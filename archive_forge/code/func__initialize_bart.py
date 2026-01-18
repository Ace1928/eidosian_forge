import os
import torch
from typing import Optional, Dict, Any
from parlai.agents.bart.convert_fairseq_to_parlai import ConversionScript
from parlai.agents.bart.modules import BartModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import compare_init_model_opts
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, History, TorchAgent
from parlai.utils.typing import TShared
from parlai.zoo.bart.build import download, CONVERSION_ARGS, BART_ARGS
def _initialize_bart(self, opt: Opt) -> Opt:
    """
        Download and convert BART pre-trained models.

        Additionally, convert `init-fairseq-model` if necessary.

        :param opt:
            ParlAI-parsed options

        :return opt:
            return opt with BART-specific args.
        """
    if not opt.get('converting'):
        download(opt['datapath'])
        opt['init_model'] = os.path.join(opt['datapath'], 'models/bart/bart_large/model')
    if opt.get('init_fairseq_model'):
        opt = self._convert_model(opt)
    opt.update(BART_ARGS)
    compare_init_model_opts(opt, opt)
    return opt