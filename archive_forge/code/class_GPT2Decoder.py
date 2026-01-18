from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor
import torch
class GPT2Decoder(torch.nn.Module):
    """
    GPT2 Decoder.

    This decoder is initialized with the pretrained model from Hugging Face.
    """

    def __init__(self, opt, dict):
        super().__init__()
        model_sz = opt['gpt2_size']
        fle_key = 'gpt2' if model_sz == 'small' else f'gpt2-{model_sz}'
        self.transformer = GPT2Model.from_pretrained(fle_key)
        self.start_idx = dict.start_idx
        self.null_idx = dict.null_idx
        self.add_start_token = False
        if opt['add_special_tokens']:
            self.transformer.resize_token_embeddings(len(dict.tokenizer))
            self.add_start_token = opt['add_start_token']
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()

    def forward(self, input, encoder_state, incr_state=None):
        attention_mask = None
        if incr_state is None:
            if not self.add_start_token and input.size(1) == 1 and (int(input[0][0]) == self.start_idx):
                model_input = encoder_state
            else:
                model_input, _ = concat_without_padding(encoder_state, input, use_cuda=self.use_cuda, null_idx=self.null_idx)
                attention_mask = model_input != self.null_idx
        else:
            model_input = input[:, -1].unsqueeze(1)
        transformer_outputs = self.transformer(model_input, past=incr_state, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]
        return (hidden_states, new_incr_state)