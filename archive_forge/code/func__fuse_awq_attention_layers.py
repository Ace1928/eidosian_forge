from ..activations import ACT2FN
from ..modeling_utils import PreTrainedModel
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion
def _fuse_awq_attention_layers(model, module, modules_to_fuse, current_module_name, target_cls):
    """
    Fuse the Attention layers into a target class using autoawq

    Args:
        model (`~PreTrainedModel`):
            The input pretrained model
        module (`nn.Module`):
            The pytorch parent module that has layernorm modules to fuse
        modules_to_fuse (`List[str]`):
            The module fusing mapping. The dictionary has to contain a field `attention` with attention module names
            in the correct order: q, k, v, o layer
        current_module_name (`str`):
            The current submodule name
        target_cls (`~autoawq.QuantAttentionFused`):
            The `QuantAttentionFused` class as it only supports that class
            for now.
    """
    from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
    if len(modules_to_fuse['attention']) == 0:
        return
    if hasattr(module, modules_to_fuse['attention'][0]):
        q_proj = getattr(module, modules_to_fuse['attention'][0])
        if isinstance(q_proj, WQLinear_GEMV):
            linear_target_cls = WQLinear_GEMV
            cat_dim = 0
        elif isinstance(q_proj, WQLinear_GEMM):
            linear_target_cls = WQLinear_GEMM
            cat_dim = 1
        else:
            raise ValueError('Unsupported q_proj type: {type(q_proj)}')
        previous_device = q_proj.qweight.device
        k_proj = getattr(module, modules_to_fuse['attention'][1])
        v_proj = getattr(module, modules_to_fuse['attention'][2])
        o_proj = getattr(module, modules_to_fuse['attention'][3])
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None
        qkv_layer = linear_target_cls(q_proj.w_bit, q_proj.group_size, q_proj.in_features, q_proj.out_features + k_proj.out_features + v_proj.out_features, q_proj.bias is not None, next(iter(module.state_dict().values())).device)
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=cat_dim)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=cat_dim)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=cat_dim)
        if isinstance(qkv_layer, WQLinear_GEMV):
            qkv_layer.split_k_iters = q_proj.split_k_iters
        qkv_layer.bias = bias
        fused_attention_layer = target_cls(modules_to_fuse['hidden_size'], modules_to_fuse['num_attention_heads'], modules_to_fuse['num_key_value_heads'], qkv_layer, o_proj, previous_device, modules_to_fuse['max_seq_len'], use_alibi=modules_to_fuse['use_alibi'], rope_theta=modules_to_fuse.get('rope_theta', 10000.0))
        fused_attention_layer.is_hf_transformers = True
        parent_name, child_name = current_module_name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, fused_attention_layer.to(previous_device))
        del q_proj, k_proj, v_proj, o_proj