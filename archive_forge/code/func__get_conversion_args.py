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
def _get_conversion_args(self, opt: Opt) -> Dict[str, Any]:
    """
        Get args for fairseq model conversion.

        :param opt:
            ParlAI Opt

        :return args:
            returns dictionary of args to send to conversion script.
        """
    model_name = os.path.split(opt['init_fairseq_model'])[-1]
    args = CONVERSION_ARGS.copy()
    args['input'] = [opt['init_fairseq_model']]
    if opt.get('model_file') and (not os.path.exists(opt['model_file'])):
        args['output'] = opt['model_file']
    elif opt.get('output_conversion_path'):
        args['output'] = opt['output_conversion_path']
    else:
        args['output'] = os.path.join(opt['datapath'], 'models/converted_fairseq_models/', model_name)
    return args