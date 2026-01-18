from typing import List, Optional, Union
from thinc.api import Model
from thinc.types import Floats2d
from spacy.attrs import intify_attr
from spacy.errors import Errors
from spacy.tokens import Doc
from spacy.util import registry
def MultiHashEmbed_v1(width: int, attrs: List[Union[str, int]], rows: List[int], include_static_vectors: bool) -> Model[List[Doc], List[Floats2d]]:
    """Construct an embedding layer that separately embeds a number of lexical
    attributes using hash embedding, concatenates the results, and passes it
    through a feed-forward subnetwork to build a mixed representation.

    The features used can be configured with the 'attrs' argument. The suggested
    attributes are NORM, PREFIX, SUFFIX and SHAPE. This lets the model take into
    account some subword information, without constructing a fully character-based
    representation. If pretrained vectors are available, they can be included in
    the representation as well, with the vectors table will be kept static
    (i.e. it's not updated).

    The `width` parameter specifies the output width of the layer and the widths
    of all embedding tables. If static vectors are included, a learned linear
    layer is used to map the vectors to the specified width before concatenating
    it with the other embedding outputs. A single Maxout layer is then used to
    reduce the concatenated vectors to the final width.

    The `rows` parameter controls the number of rows used by the `HashEmbed`
    tables. The HashEmbed layer needs surprisingly few rows, due to its use of
    the hashing trick. Generally between 2000 and 10000 rows is sufficient,
    even for very large vocabularies. A number of rows must be specified for each
    table, so the `rows` list must be of the same length as the `attrs` parameter.

    width (int): The output width. Also used as the width of the embedding tables.
        Recommended values are between 64 and 300.
    attrs (list of attr IDs): The token attributes to embed. A separate
        embedding table will be constructed for each attribute.
    rows (List[int]): The number of rows in the embedding tables. Must have the
        same length as attrs.
    include_static_vectors (bool): Whether to also use static word vectors.
        Requires a vectors table to be loaded in the Doc objects' vocab.
    """
    HashEmbed = registry.get('layers', 'HashEmbed.v1')
    FeatureExtractor = registry.get('layers', 'spacy.FeatureExtractor.v1')
    Maxout = registry.get('layers', 'Maxout.v1')
    StaticVectors = registry.get('layers', 'spacy.StaticVectors.v1')
    chain = registry.get('layers', 'chain.v1')
    concatenate = registry.get('layers', 'concatenate.v1')
    list2ragged = registry.get('layers', 'list2ragged.v1')
    with_array = registry.get('layers', 'with_array.v1')
    ragged2list = registry.get('layers', 'ragged2list.v1')
    if len(rows) != len(attrs):
        raise ValueError(f'Mismatched lengths: {len(rows)} vs {len(attrs)}')
    seed = 7

    def make_hash_embed(index):
        nonlocal seed
        seed += 1
        return HashEmbed(width, rows[index], column=index, seed=seed, dropout=0.0)
    embeddings = [make_hash_embed(i) for i in range(len(attrs))]
    concat_size = width * (len(embeddings) + include_static_vectors)
    if include_static_vectors:
        model = chain(concatenate(chain(FeatureExtractor(attrs), list2ragged(), with_array(concatenate(*embeddings))), StaticVectors(width, dropout=0.0)), with_array(Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True)), ragged2list())
    else:
        model = chain(FeatureExtractor(list(attrs)), list2ragged(), with_array(concatenate(*embeddings)), with_array(Maxout(width, concat_size, nP=3, dropout=0.0, normalize=True)), ragged2list())
    return model