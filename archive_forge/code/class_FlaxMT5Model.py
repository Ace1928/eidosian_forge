import jax.numpy as jnp
from ...utils import logging
from ..t5.modeling_flax_t5 import FlaxT5EncoderModel, FlaxT5ForConditionalGeneration, FlaxT5Model
from .configuration_mt5 import MT5Config
class FlaxMT5Model(FlaxT5Model):
    """
    This class overrides [`FlaxT5Model`]. Please check the superclass for the appropriate documentation alongside usage
    examples.

    Examples:

    ```python
    >>> from transformers import FlaxMT5Model, AutoTokenizer

    >>> model = FlaxMT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="np")

    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    model_type = 'mt5'
    config_class = MT5Config