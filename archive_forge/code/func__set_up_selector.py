from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def _set_up_selector(self, opt):
    selector_opt = {'datapath': opt['datapath'], 'model_file': opt['selector_model_file'], 'eval_candidates': 'inline', 'model': 'transformer/biencoder', 'batchsize': 1, 'interactive_mode': True, 'interactive_candidates': 'inline', 'override': {'model': 'transformer/biencoder', 'batchsize': 1}}
    for k, v in self.opt.items():
        if k not in selector_opt:
            selector_opt[k] = v
    self.selector = create_agent(selector_opt)