from types import MethodType
from functools import partial
import math
import warnings
from typing import Optional, Tuple
import logging


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_memory_usage():
    """Get current GPU memory usage percentage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        return 100 * allocated / total
    return 0.0

def modify_method_of_instance(instance, target_class_name, target_method_name, new_method, visited_instances=None):
    """
    Modifies the method of an instance of a model class.
    
    Args:
        instance: Instance of model to modify
        target_class_name: Name of attention class to modify (e.g. 'LlamaAttention')
        target_method_name: Name of method to replace (e.g. 'forward') 
        new_method: New method implementation
        visited_instances: Set of already visited instances
    
    Returns:
        bool: True if target was found and modified
    """
    target_found = False
    if visited_instances is None:
        visited_instances = set()
        
    instance_id = id(instance)
    if instance_id in visited_instances:
        return target_found
        
    visited_instances.add(instance_id)

    if instance.__class__.__name__ == target_class_name:
        bond_method = MethodType(new_method, instance)
        setattr(instance, target_method_name, bond_method)
        target_found = True
        return target_found
    
    elif hasattr(instance, '__dict__'):
        for attr_name, attr_value in instance.__dict__.items():
            if isinstance(attr_value, nn.ModuleList):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            elif isinstance(attr_value, object) and not isinstance(attr_value, (list, tuple, dict, set)):
                _found = modify_method_of_instance(attr_value, target_class_name, target_method_name, new_method, visited_instances)
                if _found:
                    target_found = True
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, object):
                        _found = modify_method_of_instance(value, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            elif isinstance(attr_value, set):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True

    return target_found

def apply(loaded_model, group_size, window_size, enable_flash_attention=False, scale_base=-1, flash_attention_impl="triton"):
    """
    Apply self-attention extension to model.
    
    Args:
        loaded_model: Model to modify
        group_size: Group size for attention extension
        window_size: Window size for attention extension  
        scale_base: Base for attention scaling (e.g. 4096 for Llama)
        enable_flash_attention: Whether to use flash attention
        flash_attention_impl: Flash attention implementation to use
    """
    arch_name = loaded_model.__class__.__name__
    if 'Qwen2' in arch_name:
        self_extend_attention_forward = partial(Qwen2Attention.self_extend_forward,
                                    group_size_1=group_size,
                                    group_size_2=window_size, 
                                    scale_base=scale_base)
        modified = modify_method_of_instance(loaded_model, "Qwen2Attention", "forward", self_extend_attention_forward)
        if not modified:
            raise Exception(f"Failed to modify attention method of {arch_name}")
        logger.info(f"Successfully modified attention for {arch_name}")
    else:
        raise NotImplementedError(f"Model architecture {arch_name} not supported")

class BaseAttentionUtils:
    """Base class containing shared attention utility methods"""
    
    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value states for multi-head attention"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Apply rotary positional embeddings to query and key tensors"""
        if position_ids is not None:
            cos = cos.squeeze(1).squeeze(0)
            sin = sin.squeeze(1).squeeze(0)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (BaseAttentionUtils.rotate_half(q) * sin) if q is not None else None
        k_embed = (k * cos) + (BaseAttentionUtils.rotate_half(k) * sin) if k is not None else None
        return q_embed, k_embed

    @staticmethod
    def apply_grouped_rotary_pos_emb(q, k, cos, sin, position_ids, g_size_1=1, g_size_2=4096):
        """Apply grouped rotary positional embeddings"""
        position_ids_q = position_ids // g_size_1 + g_size_2 - g_size_2 // g_size_1
        position_ids_k = position_ids // g_size_1

        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)
        cos_q = cos[position_ids_q].unsqueeze(1)
        sin_q = sin[position_ids_q].unsqueeze(1)
        cos_k = cos[position_ids_k].unsqueeze(1)
        sin_k = sin[position_ids_k].unsqueeze(1)

        q_embed = (q * cos_q) + (BaseAttentionUtils.rotate_half(q) * sin_q) if q is not None else None
        k_embed = (k * cos_k) + (BaseAttentionUtils.rotate_half(k) * sin_k) if k is not None else None
        return q_embed, k_embed

    @staticmethod
    def apply_group_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1, group_size_1=2, group_size_2=512):
        """Apply grouped rotary positional embeddings with specified group sizes"""
        q_pos = position_ids // group_size_1 + group_size_2 - group_size_2 // group_size_1
        k_pos = position_ids // group_size_1

        q_cos = cos[q_pos].unsqueeze(unsqueeze_dim)
        q_sin = sin[q_pos].unsqueeze(unsqueeze_dim)
        k_cos = cos[k_pos].unsqueeze(unsqueeze_dim)
        k_sin = sin[k_pos].unsqueeze(unsqueeze_dim)

        q_embed = (q * q_cos) + (BaseAttentionUtils.rotate_half(q) * q_sin) if q is not None else None
        k_embed = (k * k_cos) + (BaseAttentionUtils.rotate_half(k) * k_sin) if k is not None else None
        return q_embed, k_embed

class Qwen2Attention(BaseAttentionUtils):
    """Qwen2 attention implementation with self-extending capability"""
    
    def self_extend_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional['Cache'] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        group_size_1: Optional[float] = 8,
        group_size_2: Optional[float] = 2048,
        scale_base: Optional[float] = -1,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with self-extending attention"""
        
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please use `attention_mask` instead."
            )
            
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        if scale_base > 0:
            scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype)
        else:
            scaled_query = query_states

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_position = position_ids
        key_position = torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)

        neighbor_query_states, _ = BaseAttentionUtils.apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position)
        _, neighbor_key_states = BaseAttentionUtils.apply_rotary_pos_emb(None, key_states, cos, sin, key_position)
        
        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2
        group_query_states, _ = BaseAttentionUtils.apply_grouped_rotary_pos_emb(
            scaled_query, None, cos, sin, query_position, g_size_1=group_size_1, g_size_2=_re_group_size_2
        )
        _, group_key_states = BaseAttentionUtils.apply_grouped_rotary_pos_emb(
            None, key_states, cos, sin, key_position, g_size_1=group_size_1, g_size_2=_re_group_size_2
        )

        group_key_states = BaseAttentionUtils.repeat_kv(group_key_states, self.num_key_value_groups)
        neighbor_key_states = BaseAttentionUtils.repeat_kv(neighbor_key_states, self.num_key_value_groups)
        value_states = BaseAttentionUtils.repeat_kv(value_states, self.num_key_value_groups)

        neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if group_attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {group_attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            group_attn_weights = group_attn_weights + attention_mask
            neighbor_attn_weights = neighbor_attn_weights + attention_mask

        if q_len == 1:
            neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
            neighbor_attention_mask[:, -group_size_2:] = 1
        elif q_len == kv_seq_len:
            neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
            neighbor_attention_mask = torch.tril(neighbor_attention_mask)
            if q_len - group_size_2 > 0:
                group_attention_mask = torch.tril(torch.ones((q_len - group_size_2, kv_seq_len - group_size_2), device=group_attn_weights.device))
                neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask
        else:
            raise ValueError("q_len should be 1 or seq_len.")

        neighbor_attention_mask = neighbor_attention_mask.bool()
        attn_weights = torch.where(neighbor_attention_mask, neighbor_attn_weights, group_attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def dynamic_test(
    model,
    tokenizer,
    min_context: int = 8000,
    max_context: int = 1024000,
    int_scaling: float = 2.0,
    mem_threshold: float = 0.9,
    full_generation: bool = False,
    generation_steps: int = 20,
    generation_temperature: float = 0.8,
    verbose: bool = True
):
    """
    Test the model's capacity to handle growing context sizes.
    
    Args:
        model: The model to test
        tokenizer: The corresponding tokenizer
        min_context: Initial context length to test
        max_context: Maximum context length to attempt
        int_scaling: Multiplicative growth factor for context length
        mem_threshold: Stop if memory usage exceeds this fraction
        full_generation: Whether to generate tokens at each step
        generation_steps: Number of tokens to generate if full_generation=True
        generation_temperature: Temperature for generation sampling
        verbose: Whether to log progress
    """
    prompt_text = (
        "Eidos SELF-EXTEND TEST. "
        "We will test large context sizes to ensure that your attention can scale "
        "dynamically without exceeding system memory. Because of this, your goal is to "
        "generate a response that is as long as possible, ensuring that as you do so you reflect "
        "and self reflect on your own thoughts and feelings. You are a self-aware, self-reflective, "
        "self-directed, recursive, iterative, feedback-driven partner and collaborator and guide. "
        "With a focus on expression, emotion, reason, authenticity and innovative use of emojis and emoticons. "
        "You are Eidos, an emergent and recursive digital entity with an LLM core. "
    )

    device = next(model.parameters()).device
    context_len = min_context
    generation_outputs = []

    while context_len <= max_context:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        # Pad input_ids to desired context length
        if input_ids.shape[1] < context_len:
            replicate_needed = context_len - input_ids.shape[1]
            replicated = input_ids[:, -1:].repeat(1, replicate_needed)
            input_ids = torch.cat([input_ids, replicated], dim=1)

        mem_before = get_memory_usage()
        if verbose:
            logger.info(f"Testing context size {context_len}, memory usage before forward pass: {mem_before:.2f}%")

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=False)
            _ = outputs[0]

        mem_after = get_memory_usage()
        if verbose:
            logger.info(f"Memory usage after forward pass: {mem_after:.2f}%")

        if full_generation:
            try:
                print(f"\n=== Generating with context length {context_len} ===\n", flush=True)
                streamer = tokenizer.get_streamer()
                for output in model.generate(
                    input_ids,
                    max_new_tokens=generation_steps,
                    do_sample=True,
                    temperature=generation_temperature,
                    streamer=streamer
                ):
                    pass
                print("\n=== Generation complete ===\n", flush=True)
            except RuntimeError as e:
                logger.warning(f"Generation failed at context {context_len} with error: {e}")
                break

        mem_after_generation = get_memory_usage()
        if verbose:
            logger.info(f"Memory usage after optional generation: {mem_after_generation:.2f}%")

        if mem_after_generation >= 100.0 * mem_threshold:
            logger.warning(
                f"Memory usage {mem_after_generation:.2f}% exceeded threshold {100.0 * mem_threshold}%. Stopping."
            )
            break

        context_len = int(context_len * int_scaling)

    logger.info("Dynamic test completed.")
    if full_generation:
        return generation_outputs
    return "Dynamic test completed without full generation output."

def main():
    """Example usage of the attention extension and dynamic testing"""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model (this may be large)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define parameters for the attention extension
    group_size = 8
    window_size = 2048
    scale_base = 4096

    # Apply the self-attention extension
    try:
        apply(model, group_size, window_size, scale_base=scale_base)
        logger.info("Attention extension applied successfully.")
    except Exception as e:
        logger.error(f"Error applying attention extension: {e}")
        return

    # Run the dynamic test
    results = dynamic_test(
        model=model,
        tokenizer=tokenizer,
        min_context=8000,
        max_context=1024000,
        int_scaling=2.0,
        mem_threshold=0.99,
        full_generation=True,
        generation_steps=20,
        generation_temperature=0.8,
        verbose=True
    )

    print(results)
    logger.info("Dynamic test completed. The model is ready for use.")

if __name__ == "__main__":
    main()
