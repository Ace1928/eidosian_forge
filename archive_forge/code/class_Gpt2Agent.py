from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor
import torch
class Gpt2Agent(TorchGeneratorAgent):
    """
    Hugging Face GPT2 Agent.

    GPT2 is a multi-layer decoder-only Transformer.
    The decoder is initialized with pretrained weights from Hugging Face.
    Read more about this model here
    <https://huggingface.co/transformers/model_doc/gpt2.html>.

    GPT2 comes in four sizes: small, medium, large, and XL. Use the
    flag `--gpt2-size` to choose the size.

    If you are finetuning the Gpt2 agent as a dialogue agent, be sure
    to run `--add-special-tokens True`. To examine the performance of the
    agent out of the box, run with `--add-special-tokens False`, and make
    sure that the batch size is 1.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Gpt2 Args')
        agent.add_argument('--gpt2-size', type=str, default='small', choices=['small', 'medium', 'large', 'xl'], help='Which size model to initialize.')
        agent.add_argument('--add-special-tokens', type='bool', default=True, help='Add special tokens (like PAD, etc.). If False, Can only use with batch size 1.')
        agent.add_argument('--add-start-token', type='bool', default=False, help='Add start tokens when finetuning.')
        argparser.set_defaults(text_truncate=768, label_truncate=256, dict_maxexs=0)
        super(Gpt2Agent, cls).add_cmdline_args(argparser)
        warn_once('WARNING: this model is in beta and the API is subject to change.')
        return agent

    def __init__(self, opt, shared=None):
        if not opt['add_special_tokens'] and opt['batchsize'] > 1:
            raise RuntimeError('If using batchsize > 1, --add-special-tokens must be True.')
        super().__init__(opt, shared)

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return Gpt2DictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return HFGPT2Model(self.opt, self.dict)

    def _model_input(self, batch):
        """
        Override to pass in text lengths.
        """
        return (batch.text_vec, batch.text_lengths)

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    def _pad_tensor(self, items):
        """
        Override to always set fp16friendly to False.
        """
        return padded_tensor(items, pad_idx=self.NULL_IDX, use_cuda=self.use_cuda, fp16friendly=False)