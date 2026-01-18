from abc import abstractmethod
from typing import Dict, List, Union
import torch
from parlai.core.message import Message
from openchat.base import BaseAgent
class ParlaiClassificationAgent(ParlaiAgent):

    @abstractmethod
    def labels(self):
        raise NotImplemented

    def predict(self, text: str, **kwargs):
        message = self.tokenizer(text)
        batch = self.model.batchify([message])
        output = self.model.score(batch)[0].tolist()
        argmax = max(range(len(output)), key=lambda i: output[i])
        return {'input': text, 'output': self.labels()[argmax]}