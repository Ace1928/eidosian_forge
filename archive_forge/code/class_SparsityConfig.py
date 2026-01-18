import random
import torch
class SparsityConfig:
    """Abstract Configuration class to store `sparsity configuration of a self attention layer`.
    It contains shared property of different block-sparse sparsity patterns. However, each class
    needs to extend it based on required property and functionality.
    """

    def __init__(self, num_heads, block_size=16, different_layout_per_head=False):
        """Initialize the Sparsity Pattern Config.
        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block_size: optional: an integer determining the block size. Current implementation of
             sparse self-attention is based on blocked sparse matrices. In which this parameter
             defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be
             assigned a different sparsity layout; default is false and this will be satisfied
             based on availability.
        """
        self.num_heads = num_heads
        self.block_size = block_size
        self.different_layout_per_head = different_layout_per_head
        self.num_layout_heads = num_heads if different_layout_per_head else 1

    def setup_layout(self, seq_len):
        """Create layout tensor for the given sequence length
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) for sparsity layout
                of all head; initialized with zero
        """
        if seq_len % self.block_size != 0:
            raise ValueError(f'Sequence Length, {seq_len}, needs to be dividable by Block size {self.block_size}!')
        num_blocks = seq_len // self.block_size
        layout = torch.zeros((self.num_heads, num_blocks, num_blocks), dtype=torch.int64)
        return layout

    def check_and_propagate_first_head_layout(self, layout):
        """If all heads require same sparsity layout, it propagate first head layout to all heads
        Arguments:
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
             sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
             layout of all head
        """
        if not self.different_layout_per_head:
            layout[1:self.num_heads, :, :] = layout[0, :, :]
        return layout