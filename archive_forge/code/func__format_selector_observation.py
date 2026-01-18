from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def _format_selector_observation(self, knowledge_no_title, episode_done=False):
    obs = {'episode_done': episode_done}
    obs['label_candidates'] = [x for x in knowledge_no_title.split('\n') if x]
    text = self.retriever_history.get('chosen_topic', '')
    if len(self.dialogue_history) > 0:
        if len(self.dialogue_history) > 1:
            text += self.dialogue_history[-2]['text']
        text += self.dialogue_history[-1]['text']
    obs['text'] = text
    return obs