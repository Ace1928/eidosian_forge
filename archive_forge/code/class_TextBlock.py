from typing import Optional
from tensorflow import nest
from autokeras.blocks import basic
from autokeras.blocks import preprocessing
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
class TextBlock(block_module.Block):
    """Block for text data.

    # Arguments
        block_type: String. 'vanilla', 'transformer', and 'ngram'. The type of Block
            to use. 'vanilla' and 'transformer' use a TextToIntSequence vectorizer,
            whereas 'ngram' uses TextToNgramVector. If unspecified, it will be tuned
            automatically.
        max_tokens: Int. The maximum size of the vocabulary.
            If left unspecified, it will be tuned automatically.
        pretraining: String. 'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, block_type: Optional[str]=None, max_tokens: Optional[int]=None, pretraining: Optional[str]=None, **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.max_tokens = max_tokens
        self.pretraining = pretraining

    def get_config(self):
        config = super().get_config()
        config.update({BLOCK_TYPE: self.block_type, MAX_TOKENS: self.max_tokens, 'pretraining': self.pretraining})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        if self.block_type is None:
            block_type = hp.Choice(BLOCK_TYPE, [VANILLA, TRANSFORMER, NGRAM, BERT])
            with hp.conditional_scope(BLOCK_TYPE, [block_type]):
                output_node = self._build_block(hp, output_node, block_type)
        else:
            output_node = self._build_block(hp, output_node, self.block_type)
        return output_node

    def _build_block(self, hp, output_node, block_type):
        max_tokens = self.max_tokens or hp.Choice(MAX_TOKENS, [500, 5000, 20000], default=5000)
        if block_type == NGRAM:
            output_node = preprocessing.TextToNgramVector(max_tokens=max_tokens).build(hp, output_node)
            return basic.DenseBlock().build(hp, output_node)
        if block_type == BERT:
            output_node = basic.BertBlock().build(hp, output_node)
        else:
            output_node = preprocessing.TextToIntSequence(max_tokens=max_tokens).build(hp, output_node)
            if block_type == TRANSFORMER:
                output_node = basic.Transformer(max_features=max_tokens + 1, pretraining=self.pretraining).build(hp, output_node)
            else:
                output_node = basic.Embedding(max_features=max_tokens + 1, pretraining=self.pretraining).build(hp, output_node)
                output_node = basic.ConvBlock().build(hp, output_node)
            output_node = reduction.SpatialReduction().build(hp, output_node)
            output_node = basic.DenseBlock().build(hp, output_node)
        return output_node