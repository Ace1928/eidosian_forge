from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.build_data import modelzoo_path
from parlai.utils.safety import OffensiveStringMatcher
from parlai.core.agents import add_datapath_and_model_args, create_agent_from_opt_file, create_agent
from openchat.base import ParlaiClassificationAgent, EncoderLM, SingleTurn
def _create_safety_model(self, custom_model_file, device):
    from parlai.core.params import ParlaiParser
    parser = ParlaiParser(False, False)
    TransformerClassifierAgent.add_cmdline_args(parser, partial_opt=None)
    parser.set_params(model='transformer/classifier', model_file=custom_model_file, print_scores=True, data_parallel=False)
    safety_opt = parser.parse_args([])
    safety_opt['override']['no_cuda'] = False if 'cuda' in device else True
    if 'cuda:' in device:
        safety_opt['override']['gpu'] = int(device.split(':')[1])
    elif 'cuda' in device:
        safety_opt['override']['gpu'] = 0
    return create_agent(safety_opt, requireModelExists=True)