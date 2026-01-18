from the base class `SymbolDoc`, and put the extra doc as the docstring
import re as _re
from .base import build_param_doc as _build_param_doc
class EmbeddingDoc(SymbolDoc):
    """
    Examples
    --------
    Assume we want to map the 26 English alphabet letters to 16-dimensional
    vectorial representations.

    >>> vocabulary_size = 26
    >>> embed_dim = 16
    >>> seq_len, batch_size = (10, 64)
    >>> input = Variable('letters')
    >>> op = Embedding(data=input, input_dim=vocabulary_size, output_dim=embed_dim,
    ...                name='embed')
    >>> SymbolDoc.get_output_shape(op, letters=(seq_len, batch_size))
    {'embed_output': (10L, 64L, 16L)}

    >>> vocab_size, embed_dim = (26, 16)
    >>> batch_size = 12
    >>> word_vecs = test_utils.random_arrays((vocab_size, embed_dim))
    >>> op = Embedding(name='embed', input_dim=vocab_size, output_dim=embed_dim)
    >>> x = np.random.choice(vocab_size, batch_size)
    >>> y = test_utils.simple_forward(op, embed_data=x, embed_weight=word_vecs)
    >>> y_np = word_vecs[x]
    >>> test_utils.almost_equal(y, y_np)
    True
    """