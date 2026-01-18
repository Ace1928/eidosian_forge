import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxWhisperTimeStampLogitsProcessor(FlaxLogitsProcessor):
    """
    Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log
    probs to `inf` so that they are sampled at their corresponding index.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
    """

    def __init__(self, generate_config, model_config, decoder_input_length):
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        self.begin_index = decoder_input_length + 1
        if generate_config.is_multilingual:
            self.begin_index += 2
        if hasattr(generate_config, 'max_initial_timestamp_index'):
            self.max_initial_timestamp_index = generate_config.max_initial_timestamp_index
        else:
            self.max_initial_timestamp_index = model_config.vocab_size
        if self.max_initial_timestamp_index is None:
            self.max_initial_timestamp_index = model_config.vocab_size

    def __call__(self, input_ids, scores, cur_len):
        scores = scores.at[:, self.no_timestamps_token_id].set(-float('inf'))

        def handle_pairs(input_ids_k, scores_k):
            last_was_timestamp = jnp.where(cur_len - self.begin_index >= 1, True, False)
            last_was_timestamp = jnp.where(input_ids_k[cur_len - 1] >= self.timestamp_begin, True and last_was_timestamp, False)
            penultimate_was_timestamp = jnp.where(cur_len - self.begin_index < 2, True, False)
            penultimate_was_timestamp = jnp.where(input_ids_k[cur_len - 2] >= self.timestamp_begin, True, penultimate_was_timestamp)
            return jnp.where(last_was_timestamp, jnp.where(penultimate_was_timestamp > 0, scores_k.at[self.timestamp_begin:].set(-float('inf')), scores_k.at[:self.eos_token_id].set(-float('inf'))), scores_k)
        scores = jax.vmap(handle_pairs)(input_ids, scores)
        apply_max_initial_timestamp = jnp.where(cur_len == self.begin_index, True, False)
        apply_max_initial_timestamp = jnp.where(self.max_initial_timestamp_index is not None, True and apply_max_initial_timestamp, False)
        last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
        scores = jnp.where(apply_max_initial_timestamp, scores.at[:, last_allowed + 1:].set(-float('inf')), scores)
        logprobs = jax.nn.log_softmax(scores, axis=-1)

        def handle_cumulative_probs(logprobs_k, scores_k):
            timestamp_logprob = jax.nn.logsumexp(logprobs_k[self.timestamp_begin:], axis=-1)
            max_text_token_logprob = jnp.max(logprobs_k[:self.timestamp_begin])
            return jnp.where(timestamp_logprob > max_text_token_logprob, scores_k.at[:self.timestamp_begin].set(-float('inf')), scores_k)
        scores = jax.vmap(handle_cumulative_probs)(logprobs, scores)
        return scores