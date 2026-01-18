import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from .pipeline_utils import DiffusionPipelineMixin, rescale_noise_cfg
def _encode_prompt(self, prompt: Union[str, List[str]], num_images_per_prompt: int, do_classifier_free_guidance: bool, negative_prompt: Optional[Union[str, list]], prompt_embeds: Optional[np.ndarray]=None, negative_prompt_embeds: Optional[np.ndarray]=None, pooled_prompt_embeds: Optional[np.ndarray]=None, negative_pooled_prompt_embeds: Optional[np.ndarray]=None):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`Union[str, List[str]]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
        """
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
    text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
    if prompt_embeds is None:
        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='np')
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding='longest', return_tensors='np').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and (not np.array_equal(text_input_ids, untruncated_ids)):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1:-1])
                logger.warning(f'The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}')
            prompt_embeds = text_encoder(input_ids=text_input_ids.astype(text_encoder.input_dtype.get('input_ids', np.int32)))
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-2]
            prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = np.concatenate(prompt_embeds_list, axis=-1)
    zero_out_negative_prompt = negative_prompt is None and self.config['force_zeros_for_empty_prompt']
    if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        negative_prompt_embeds = np.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = np.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ''
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.')
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.')
        else:
            uncond_tokens = negative_prompt
        negative_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(uncond_tokens, padding='max_length', max_length=max_length, truncation=True, return_tensors='np')
            negative_prompt_embeds = text_encoder(input_ids=uncond_input.input_ids.astype(text_encoder.input_dtype.get('input_ids', np.int32)))
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds[-2]
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)
            negative_prompt_embeds_list.append(negative_prompt_embeds)
        negative_prompt_embeds = np.concatenate(negative_prompt_embeds_list, axis=-1)
    pooled_prompt_embeds = np.repeat(pooled_prompt_embeds, num_images_per_prompt, axis=0)
    negative_pooled_prompt_embeds = np.repeat(negative_pooled_prompt_embeds, num_images_per_prompt, axis=0)
    return (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)