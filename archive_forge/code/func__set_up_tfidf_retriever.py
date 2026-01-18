from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def _set_up_tfidf_retriever(self, opt):
    retriever_opt = {'model_file': opt['retriever_model_file'], 'remove_title': False, 'datapath': opt['datapath'], 'override': {'remove_title': False}}
    self.retriever = create_agent(retriever_opt)
    self._set_up_sent_tok()
    wiki_map_path = os.path.join(self.model_path, 'chosen_topic_to_passage.json')
    self.wiki_map = json.load(open(wiki_map_path, 'r'))