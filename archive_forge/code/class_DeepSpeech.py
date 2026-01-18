import torch
class DeepSpeech(torch.nn.Module):
    """DeepSpeech architecture introduced in
    *Deep Speech: Scaling up end-to-end speech recognition* :cite:`hannun2014deep`.

    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
        n_class: Number of output classes
    """

    def __init__(self, n_feature: int, n_hidden: int=2048, n_class: int=40, dropout: float=0.0) -> None:
        super(DeepSpeech, self).__init__()
        self.n_hidden = n_hidden
        self.fc1 = FullyConnected(n_feature, n_hidden, dropout)
        self.fc2 = FullyConnected(n_hidden, n_hidden, dropout)
        self.fc3 = FullyConnected(n_hidden, n_hidden, dropout)
        self.bi_rnn = torch.nn.RNN(n_hidden, n_hidden, num_layers=1, nonlinearity='relu', bidirectional=True)
        self.fc4 = FullyConnected(n_hidden, n_hidden, dropout)
        self.out = torch.nn.Linear(n_hidden, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
        Returns:
            Tensor: Predictor tensor of dimension (batch, time, class).
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.squeeze(1)
        x = x.transpose(0, 1)
        x, _ = self.bi_rnn(x)
        x = x[:, :, :self.n_hidden] + x[:, :, self.n_hidden:]
        x = self.fc4(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x