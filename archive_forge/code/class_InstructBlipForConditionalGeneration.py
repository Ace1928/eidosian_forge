import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_instructblip import InstructBlipConfig, InstructBlipQFormerConfig, InstructBlipVisionConfig
@add_start_docstrings('\n    InstructBLIP Model for generating text given an image and an optional text prompt. The model consists of a vision\n    encoder, Querying Transformer (Q-Former) and a language model.\n\n    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue\n    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.\n    ', INSTRUCTBLIP_START_DOCSTRING)
class InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        self.vision_model = InstructBlipVisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)
        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)
        self.language_model = language_model
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        """
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map
        if len(hf_device_map) > 1 and 'language_model' not in hf_device_map and (torch.cuda.device_count() > 1):
            logger.warning('The `language_model` is not in the `hf_device_map` dictionary and you are running your script in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`. Please pass a `device_map` that contains `language_model` to remove this warning. Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for more details on creating a `device_map` for large models.')
        if hasattr(self.language_model, '_hf_hook'):
            self.language_model._hf_hook.io_same_device = True

    @add_start_docstrings_to_model_forward(INSTRUCTBLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=InstructBlipForConditionalGenerationModelOutput, config_class=InstructBlipVisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, qformer_input_ids: torch.FloatTensor, qformer_attention_mask: Optional[torch.LongTensor]=None, input_ids: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(input_ids=qformer_input_ids, attention_mask=qformer_attention_mask, query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        query_output = query_outputs[0][:, :query_tokens.size(1), :]
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)
        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss_fct = CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, labels=labels)
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]
        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return (loss,) + output if loss is not None else output
        return InstructBlipForConditionalGenerationModelOutput(loss=loss, logits=logits, vision_outputs=vision_outputs, qformer_outputs=query_outputs, language_model_outputs=outputs)

    @torch.no_grad()
    def generate(self, pixel_values: torch.FloatTensor, qformer_input_ids: Optional[torch.LongTensor]=None, qformer_attention_mask: Optional[torch.LongTensor]=None, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, **generate_kwargs) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt to be fed to the Q-Former module.
            qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, 'hf_device_map'):
            self._preprocess_accelerate()
        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(input_ids=qformer_input_ids, attention_mask=qformer_attention_mask, query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=True)
        query_output = query_outputs.last_hidden_state[:, :query_tokens.size(1), :]
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)
        if input_ids is None:
            input_ids = torch.LongTensor([[self.config.text_config.bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        outputs = self.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
        if self.config.text_config.architectures[0] == 'LLaMAForCausalLM':
            if isinstance(outputs, torch.Tensor):
                outputs[outputs == 0] = 2
            else:
                outputs.sequences[outputs.sequences == 0] = 2
        return outputs