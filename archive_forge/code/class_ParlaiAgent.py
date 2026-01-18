from abc import abstractmethod
from typing import Dict, List, Union
import torch
from parlai.core.message import Message
from openchat.base import BaseAgent
class ParlaiAgent(BaseAgent):

    def __init__(self, name, suffix, device, maxlen, model):
        super(ParlaiAgent, self).__init__(name=name, suffix=suffix, device=device, maxlen=maxlen, model=model, tokenizer=self.tokenizer)
        if 'cuda:' in device:
            self.model.opt['gpu'] = int(device.split(':')[1])
        elif 'cuda' in device:
            self.model.opt['gpu'] = 0

    def tokenizer(self, message: Union[str, List[str]], padding=False):
        if isinstance(message, str):
            return {'input_ids': self.model.dict.txt2vec(message)}
        elif isinstance(message, list):
            if all((isinstance(s, str) for s in message)):
                tokens = [self.model.dict.txt2vec(s) for s in message]
                if padding:
                    tokens = self.model._pad_tensor(tokens)[0]
                return {'input_ids': tokens}
            else:
                raise TypeError(f'type error: {type(message)}, input type must be one of [str, List[str]]')
        else:
            raise TypeError(f'type error: {type(message)}, input type must be one of [str, List[str]]')