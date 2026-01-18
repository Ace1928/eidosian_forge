import collections
import gzip
import io
import logging
import struct
import numpy as np
def _load_vocab(fin, new_format, encoding='utf-8'):
    """Load a vocabulary from a FB binary.

    Before the vocab is ready for use, call the prepare_vocab function and pass
    in the relevant parameters from the model.

    Parameters
    ----------
    fin : file
        An open file pointer to the binary.
    new_format: boolean
        True if the binary is of the newer format.
    encoding : str
        The encoding to use when decoding binary data into words.

    Returns
    -------
    tuple
        The loaded vocabulary.  Keys are words, values are counts.
        The vocabulary size.
        The number of words.
        The number of tokens.
    """
    vocab_size, nwords, nlabels = _struct_unpack(fin, '@3i')
    if nlabels > 0:
        raise NotImplementedError('Supervised fastText models are not supported')
    logger.info('loading %s words for fastText model from %s', vocab_size, fin.name)
    ntokens = _struct_unpack(fin, '@q')[0]
    if new_format:
        pruneidx_size, = _struct_unpack(fin, '@q')
    raw_vocab = collections.OrderedDict()
    for i in range(vocab_size):
        word_bytes = io.BytesIO()
        char_byte = fin.read(1)
        while char_byte != _END_OF_WORD_MARKER:
            word_bytes.write(char_byte)
            char_byte = fin.read(1)
        word_bytes = word_bytes.getvalue()
        try:
            word = word_bytes.decode(encoding)
        except UnicodeDecodeError:
            word = word_bytes.decode(encoding, errors='backslashreplace')
            logger.error('failed to decode invalid unicode bytes %r; replacing invalid characters, using %r', word_bytes, word)
        count, _ = _struct_unpack(fin, '@qb')
        raw_vocab[word] = count
    if new_format:
        for j in range(pruneidx_size):
            _struct_unpack(fin, '@2i')
    return (raw_vocab, vocab_size, nwords, ntokens)