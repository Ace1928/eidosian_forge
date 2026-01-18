from parlai.agents.transformer.transformer import TransformerRankerAgent
from .wizard_dict import WizardDictAgent
import numpy as np
import torch
def _vectorize_memories(self, observation):
    """
        Override abstract method from TransformerRankerAgent to use knowledge field as
        memories.
        """
    if not self.use_knowledge:
        return observation
    observation['memory_vecs'] = []
    checked = observation.get('checked_sentence', '')
    if observation.get('knowledge'):
        knowledge = observation['knowledge'].split('\n')[:-1]
    else:
        knowledge = []
    to_vectorize = []
    if checked and self.chosen_sentence:
        to_vectorize = [checked]
    elif (self.knowledge_dropout == 0 or not self.is_training) and knowledge:
        to_vectorize = knowledge
    elif knowledge:
        for line in knowledge:
            if checked and checked in line:
                keep = 1
            else:
                keep = np.random.binomial(1, 1 - self.knowledge_dropout)
            if keep:
                to_vectorize.append(line)
    observation.force_set('memory_vecs', [self._vectorize_text(line, truncate=self.knowledge_truncate) for line in to_vectorize])
    return observation