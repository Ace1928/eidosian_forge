import random
import torch
class DenseSparsityConfig(SparsityConfig):
    """Configuration class to store `Dense` configuration.
    In reality, this is not sparse and all blocks are used. We keep it for the sake of comparison and
    comprehension.
    """

    def __init__(self, num_heads, block_size=16, different_layout_per_head=False):
        """Initialize the Dense Sparsity Pattern Config.
        In reality, this is not sparse and all blocks are used. We keep it for the sake of comparison
        and comprehension.
        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block_size: optional: an integer determining the block size. Current implementation of
             sparse self-attention is based on blocked sparse matrices. In which this parameter
             defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: this is just for the sake of consistency with
             other sparsity formats; can ignore it for DenseSparsityConfig
        """
        super().__init__(num_heads, block_size, different_layout_per_head)

    def make_layout(self, seq_len):
        """Set 1 to all blocks of the layout meanins the pattern is dense; not sparse.
        Arguments:
             seq_len: required: an integer determining the underling sequence length;
             must be <= max sequence length
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
             layout of all head; for dense everything is 1
        """
        layout = self.setup_layout(seq_len)
        layout[:, :, :] = 1
        return layout