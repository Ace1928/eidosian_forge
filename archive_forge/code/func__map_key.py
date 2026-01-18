import re
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model
def _map_key(key):
    key_ = key
    if key.startswith('w2v_model.'):
        key = key.replace('w2v_model.', '')
    if re.match('(mask_emb|quantizer|project_q|final_proj|mask_emb)', key):
        return None
    match = re.match('feature_extractor\\.conv_layers\\.0\\.2\\.(weight|bias)', key)
    if match:
        return f'feature_extractor.conv_layers.0.layer_norm.{match.group(1)}'
    match = re.match('feature_extractor\\.conv_layers\\.(\\d+)\\.0\\.(weight|bias)', key)
    if match:
        return f'feature_extractor.conv_layers.{match.group(1)}.conv.{match.group(2)}'
    match = re.match('feature_extractor\\.conv_layers\\.(\\d+)\\.2\\.1\\.(weight|bias)', key)
    if match:
        return f'feature_extractor.conv_layers.{match.group(1)}.layer_norm.{match.group(2)}'
    match = re.match('post_extract_proj\\.(weight|bias)', key)
    if match:
        return f'encoder.feature_projection.projection.{match.group(1)}'
    match = re.match('layer_norm\\.(weight|bias)', key)
    if match:
        return f'encoder.feature_projection.layer_norm.{match.group(1)}'
    match = re.match('encoder\\.pos_conv\\.0\\.(bias|weight_g|weight_v)', key)
    if match:
        return f'encoder.transformer.pos_conv_embed.conv.{match.group(1)}'
    match = re.match('encoder\\.layer_norm\\.(weight|bias)', key)
    if match:
        return f'encoder.transformer.layer_norm.{match.group(1)}'
    match = re.match('encoder\\.layers\\.(\\d+)\\.self_attn\\.((k_|v_|q_|out_)proj\\.(weight|bias))', key)
    if match:
        return f'encoder.transformer.layers.{match.group(1)}.attention.{match.group(2)}'
    match = re.match('encoder\\.layers\\.(\\d+)\\.self_attn_layer_norm\\.(weight|bias)', key)
    if match:
        return f'encoder.transformer.layers.{match.group(1)}.layer_norm.{match.group(2)}'
    match = re.match('encoder\\.layers\\.(\\d+)\\.fc1\\.(weight|bias)', key)
    if match:
        return f'encoder.transformer.layers.{match.group(1)}.feed_forward.intermediate_dense.{match.group(2)}'
    match = re.match('encoder\\.layers\\.(\\d+)\\.fc2\\.(weight|bias)', key)
    if match:
        return f'encoder.transformer.layers.{match.group(1)}.feed_forward.output_dense.{match.group(2)}'
    match = re.match('encoder\\.layers\\.(\\d+)\\.final_layer_norm\\.(weight|bias)', key)
    if match:
        return f'encoder.transformer.layers.{match.group(1)}.final_layer_norm.{match.group(2)}'
    match = re.match('proj\\.(weight|bias)', key)
    if match:
        return f'aux.{match.group(1)}'
    if key in ['label_embs_concat']:
        return key
    raise ValueError(f'Unexpected key: {key_}')