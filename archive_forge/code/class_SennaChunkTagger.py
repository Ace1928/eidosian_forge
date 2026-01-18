from nltk.classify import Senna
class SennaChunkTagger(Senna):

    def __init__(self, path, encoding='utf-8'):
        super().__init__(path, ['chk'], encoding)

    def tag_sents(self, sentences):
        """
        Applies the tag method over a list of sentences. This method will return
        for each sentence a list of tuples of (word, tag).
        """
        tagged_sents = super().tag_sents(sentences)
        for i in range(len(tagged_sents)):
            for j in range(len(tagged_sents[i])):
                annotations = tagged_sents[i][j]
                tagged_sents[i][j] = (annotations['word'], annotations['chk'])
        return tagged_sents

    def bio_to_chunks(self, tagged_sent, chunk_type):
        """
        Extracts the chunks in a BIO chunk-tagged sentence.

        >>> from nltk.tag import SennaChunkTagger
        >>> chktagger = SennaChunkTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
        >>> sent = 'What is the airspeed of an unladen swallow ?'.split()
        >>> tagged_sent = chktagger.tag(sent)  # doctest: +SKIP
        >>> tagged_sent  # doctest: +SKIP
        [('What', 'B-NP'), ('is', 'B-VP'), ('the', 'B-NP'), ('airspeed', 'I-NP'),
        ('of', 'B-PP'), ('an', 'B-NP'), ('unladen', 'I-NP'), ('swallow', 'I-NP'),
        ('?', 'O')]
        >>> list(chktagger.bio_to_chunks(tagged_sent, chunk_type='NP'))  # doctest: +SKIP
        [('What', '0'), ('the airspeed', '2-3'), ('an unladen swallow', '5-6-7')]

        :param tagged_sent: A list of tuples of word and BIO chunk tag.
        :type tagged_sent: list(tuple)
        :param tagged_sent: The chunk tag that users want to extract, e.g. 'NP' or 'VP'
        :type tagged_sent: str

        :return: An iterable of tuples of chunks that users want to extract
          and their corresponding indices.
        :rtype: iter(tuple(str))
        """
        current_chunk = []
        current_chunk_position = []
        for idx, word_pos in enumerate(tagged_sent):
            word, pos = word_pos
            if '-' + chunk_type in pos:
                current_chunk.append(word)
                current_chunk_position.append(idx)
            elif current_chunk:
                _chunk_str = ' '.join(current_chunk)
                _chunk_pos_str = '-'.join(map(str, current_chunk_position))
                yield (_chunk_str, _chunk_pos_str)
                current_chunk = []
                current_chunk_position = []
        if current_chunk:
            yield (' '.join(current_chunk), '-'.join(map(str, current_chunk_position)))