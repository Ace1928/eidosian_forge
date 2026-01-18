class DFTO(nn.Module):
    """
    Dynamic Fractal Topology Output (DFTO) class.
    This class defines a neural network module that applies various aggregation operations on the input tensor.
    """

    def __init__(self, operation: str = "sum"):
        """
        Initialize the DFTO module.

        Parameters:
        operation (str): The aggregation operation to apply. Options are 'sum', 'max', 'avg', 'min', 'aggregate', 'power_spectrum'.
        """
        super(DFTO, self).__init__()
        self.operation = operation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DFTO module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the specified aggregation operation.
        """
        x = x.float()  # Ensure input is float
        if self.operation == "sum":
            return torch.sum(x, dim=1)
        elif self.operation == "max":
            return torch.max(x, dim=1)[0]
        elif self.operation == "avg":
            return torch.mean(x, dim=1)
        elif self.operation == "min":
            return torch.min(x, dim=1)[0]
        elif self.operation == "aggregate":
            return torch.sum(x, dim=1) / x.size(1)
        elif self.operation == "power_spectrum":
            return torch.sum(x**2, dim=1)
