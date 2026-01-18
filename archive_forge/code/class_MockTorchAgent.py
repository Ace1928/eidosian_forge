from parlai.core.torch_agent import TorchAgent, Output
import torch
from parlai.core.agents import Agent
class MockTorchAgent(TorchAgent):
    """
    Use MockDict instead of regular DictionaryAgent.
    """

    @staticmethod
    def dictionary_class():
        """
        Replace normal dictionary class with mock one.
        """
        return MockDict

    def __init__(self, opt, shared=None):
        self.model = self.build_model()
        self.criterion = self.build_criterion()
        super().__init__(opt, shared)

    def build_model(self):
        return torch.nn.Module()

    def build_criterion(self):
        return torch.nn.NLLLoss()

    def train_step(self, batch):
        """
        Return confirmation of training.
        """
        return Output(['Training {}!'.format(i) for i in range(len(batch.text_vec))])

    def eval_step(self, batch):
        """
        Return confirmation of evaluation.
        """
        return Output(['Evaluating {} (responding to {})!'.format(i, batch.observations[i]['text']) for i in range(len(batch.text_vec))])