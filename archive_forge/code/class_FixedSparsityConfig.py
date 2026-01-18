import random
import torch
class FixedSparsityConfig(SparsityConfig):
    """Configuration class to store `Fixed` sparsity configuration.
    For more details about this sparsity config, please see `Generative Modeling with
    Sparse Transformers`: https://arxiv.org/abs/1904.10509; this has been customized.
    This class extends parent class of `SparsityConfig` and customizes it for `Fixed` sparsity.
    """

    def __init__(self, num_heads, block_size=16, different_layout_per_head=False, num_local_blocks=4, num_global_blocks=1, attention='bidirectional', horizontal_global_attention=False, num_different_global_patterns=1):
        """Initialize `Fixed` Sparsity Pattern Config.
        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial
        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             block_size: optional: an integer determining the block size. Current implementation of
                sparse self-attention is based on blocked sparse matrices. In which this parameter
                defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be
                assigned a different sparsity layout; default is false and this will be satisfied
                based on availability.
             num_local_blocks: optional: an integer determining the number of blocks in local attention
                window.
             num_global_blocks: optional: an integer determining how many consecutive blocks in a local
                window is used as the representative of the window for global attention.
             attention: optional: a string determining attention type. Attention can be `unidirectional`,
                such as autoregressive models, in which tokens attend only to tokens appear before them
                in the context. Considering that, the upper triangular of attention matrix is empty as
                above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to
                any other tokens before or after them. Then, the upper triangular part of the attention
                matrix is mirror of the lower triangular in the above figure.
             horizontal_global_attention: optional: a boolean determining if blocks that are global
                representative of a local window, also attend to all other blocks. This is valid only if
                attention type is `bidirectional`. Looking at the attention matrix, that means global
                attention not only includes the vertical blocks, but also horizontal blocks.
             num_different_global_patterns: optional: an integer determining number of different global
                attentions layouts. While global attention can be fixed by which block/s are representative
                of any local window, since there are multi-heads, each head can use a different global representative.
                For example, with 4 blocks local window and global attention size of 1 block, we can have 4 different
                versions in which the first, Second, third, or forth block of each local window can be global
                representative of that window. This parameter determines how many of such patterns we want.
                Of course, there is a limitation based on num_local_blocks and num_global_blocks.
        """
        super().__init__(num_heads, block_size, different_layout_per_head)
        self.num_local_blocks = num_local_blocks
        if num_local_blocks % num_global_blocks != 0:
            raise ValueError(f'Number of blocks in a local window, {num_local_blocks},\n                    must be dividable by number of global blocks, {num_global_blocks}!')
        self.num_global_blocks = num_global_blocks
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only "uni/bi-directional" attentions are supported for now!')
        self.attention = attention
        if attention != 'bidirectional' and horizontal_global_attention:
            raise ValueError('only "bi-directional" attentions can support horizontal global attention!')
        self.horizontal_global_attention = horizontal_global_attention
        if num_different_global_patterns > 1 and (not different_layout_per_head):
            raise ValueError('Number of different layouts cannot be more than one when you have set a single layout\n                for all heads! Set different_layout_per_head to True.')
        if num_different_global_patterns > num_local_blocks // num_global_blocks:
            raise ValueError(f'Number of layout versions (num_different_global_patterns), {num_different_global_patterns},\n                cannot be larger than number of local window blocks divided by number of global blocks,\n                {num_local_blocks} / {num_global_blocks} = {num_local_blocks // num_global_blocks}!')
        self.num_different_global_patterns = num_different_global_patterns

    def set_local_layout(self, h, layout):
        """Sets local attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which local layout is set
        """
        num_blocks = layout.shape[1]
        for i in range(0, num_blocks, self.num_local_blocks):
            end = min(i + self.num_local_blocks, num_blocks)
            for row in range(i, end):
                for col in range(i, row + 1 if self.attention == 'unidirectional' else end):
                    layout[h, row, col] = 1
        return layout

    def set_global_layout(self, h, layout):
        """Sets global attention layout used by the given head in the sparse attention.
        Currently we set global blocks starting from the last block of a local window to the first one.
        That means if a local window consists of 4 blocks and global attention size is one block, we use
        block #4 in each local window as global. If we have different layout per head, then other heads
        will get #3, #2, and #1. And if we have more heads (and different layout has set) than num of global
        attentions, multiple head may have same global attentions.
        Note) if horizontal_global_attention is set, global blocks will be set both horizontally and
        vertically.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which global layout is set
        """
        num_blocks = layout.shape[1]
        first_global_block_idx = self.num_local_blocks - (1 + h % self.num_different_global_patterns) * self.num_global_blocks
        end = num_blocks - num_blocks % self.num_local_blocks
        for i in range(first_global_block_idx, end, self.num_local_blocks):
            first_row = 0 if self.attention == 'bidirectional' else i
            layout[h, first_row:, i:i + self.num_global_blocks] = 1
            if self.horizontal_global_attention:
                layout[h, i:i + self.num_global_blocks, :] = 1
        if end < num_blocks:
            start = min(end + first_global_block_idx, num_blocks - self.num_global_blocks)
            end = start + self.num_global_blocks
            first_row = 0 if self.attention == 'bidirectional' else start
            layout[h, first_row:, start:end] = 1
            if self.horizontal_global_attention:
                layout[h, start:end, :] = 1
        return layout

    def make_layout(self, seq_len):
        """Generates `Fixed` sparsity layout used by each head in the sparse attention.
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing `Fixed`
                sparsity layout of all head
        """
        layout = self.setup_layout(seq_len)
        for h in range(0, self.num_layout_heads):
            layout = self.set_local_layout(h, layout)
            layout = self.set_global_layout(h, layout)
        layout = self.check_and_propagate_first_head_layout(layout)
        return layout