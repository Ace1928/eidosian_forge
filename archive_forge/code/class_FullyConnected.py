import torch
class FullyConnected(torch.nn.Module):
    """
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
    """

    def __init__(self, n_feature: int, n_hidden: int, dropout: float, relu_max_clip: int=20) -> None:
        super(FullyConnected, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x