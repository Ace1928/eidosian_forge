from typing import List, Optional, Union
from thinc.api import Model
from thinc.types import Floats2d
from spacy.attrs import intify_attr
from spacy.errors import Errors
from spacy.tokens import Doc
from spacy.util import registry
def CharacterEmbed_v1(width: int, rows: int, nM: int, nC: int, include_static_vectors: bool, feature: Union[int, str]='LOWER') -> Model[List[Doc], List[Floats2d]]:
    """Construct an embedded representation based on character embeddings, using
    a feed-forward network. A fixed number of UTF-8 byte characters are used for
    each word, taken from the beginning and end of the word equally. Padding is
    used in the centre for words that are too short.

    For instance, let's say nC=4, and the word is "jumping". The characters
    used will be jung (two from the start, two from the end). If we had nC=8,
    the characters would be "jumpping": 4 from the start, 4 from the end. This
    ensures that the final character is always in the last position, instead
    of being in an arbitrary position depending on the word length.

    The characters are embedded in a embedding table with a given number of rows,
    and the vectors concatenated. A hash-embedded vector of the LOWER of the word is
    also concatenated on, and the result is then passed through a feed-forward
    network to construct a single vector to represent the information.

    feature (int or str): An attribute to embed, to concatenate with the characters.
    width (int): The width of the output vector and the feature embedding.
    rows (int): The number of rows in the LOWER hash embedding table.
    nM (int): The dimensionality of the character embeddings. Recommended values
        are between 16 and 64.
    nC (int): The number of UTF-8 bytes to embed per word. Recommended values
        are between 3 and 8, although it may depend on the length of words in the
        language.
    include_static_vectors (bool): Whether to also use static word vectors.
        Requires a vectors table to be loaded in the Doc objects' vocab.
    """
    CharEmbed = registry.get('layers', 'spacy.CharEmbed.v1')
    FeatureExtractor = registry.get('layers', 'spacy.FeatureExtractor.v1')
    Maxout = registry.get('layers', 'Maxout.v1')
    HashEmbed = registry.get('layers', 'HashEmbed.v1')
    StaticVectors = registry.get('layers', 'spacy.StaticVectors.v1')
    chain = registry.get('layers', 'chain.v1')
    concatenate = registry.get('layers', 'concatenate.v1')
    list2ragged = registry.get('layers', 'list2ragged.v1')
    with_array = registry.get('layers', 'with_array.v1')
    ragged2list = registry.get('layers', 'ragged2list.v1')
    feature = intify_attr(feature)
    if feature is None:
        raise ValueError(Errors.E911(feat=feature))
    if include_static_vectors:
        model = chain(concatenate(chain(CharEmbed(nM=nM, nC=nC), list2ragged()), chain(FeatureExtractor([feature]), list2ragged(), with_array(HashEmbed(nO=width, nV=rows, column=0, seed=5))), StaticVectors(width, dropout=0.0)), with_array(Maxout(width, nM * nC + 2 * width, nP=3, normalize=True, dropout=0.0)), ragged2list())
    else:
        model = chain(concatenate(chain(CharEmbed(nM=nM, nC=nC), list2ragged()), chain(FeatureExtractor([feature]), list2ragged(), with_array(HashEmbed(nO=width, nV=rows, column=0, seed=5)))), with_array(Maxout(width, nM * nC + width, nP=3, normalize=True, dropout=0.0)), ragged2list())
    return model