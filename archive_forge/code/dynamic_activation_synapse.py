class DAS(nn.Module):
    """
    Dynamic Activation Synapses (DASs) class.
    This class defines a neural network module that applies a linear transformation followed by a sigmoid activation function.
    The output is scaled by a learnable parameter.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the DAS module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        """
        super(DAS, self).__init__()
        self.basis_function = nn.Linear(input_size, output_size)
        self.param = nn.Parameter(torch.randn(output_size, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DAS module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the linear transformation, sigmoid activation, and scaling.
        """
        x = x.float()  # Ensure input is float
        return self.param * torch.sigmoid(self.basis_function(x))
