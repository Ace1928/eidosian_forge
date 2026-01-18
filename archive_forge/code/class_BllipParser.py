from nltk.parse.api import ParserI
from nltk.tree import Tree
from NLTK's downloader. More unified parsing models can be obtained with
class BllipParser(ParserI):
    """
    Interface for parsing with BLLIP Parser. BllipParser objects can be
    constructed with the ``BllipParser.from_unified_model_dir`` class
    method or manually using the ``BllipParser`` constructor.
    """

    def __init__(self, parser_model=None, reranker_features=None, reranker_weights=None, parser_options=None, reranker_options=None):
        """
        Load a BLLIP Parser model from scratch. You'll typically want to
        use the ``from_unified_model_dir()`` class method to construct
        this object.

        :param parser_model: Path to parser model directory
        :type parser_model: str

        :param reranker_features: Path the reranker model's features file
        :type reranker_features: str

        :param reranker_weights: Path the reranker model's weights file
        :type reranker_weights: str

        :param parser_options: optional dictionary of parser options, see
            ``bllipparser.RerankingParser.RerankingParser.load_parser_options()``
            for more information.
        :type parser_options: dict(str)

        :param reranker_options: optional
            dictionary of reranker options, see
            ``bllipparser.RerankingParser.RerankingParser.load_reranker_model()``
            for more information.
        :type reranker_options: dict(str)
        """
        _ensure_bllip_import_or_error()
        parser_options = parser_options or {}
        reranker_options = reranker_options or {}
        self.rrp = RerankingParser()
        self.rrp.load_parser_model(parser_model, **parser_options)
        if reranker_features and reranker_weights:
            self.rrp.load_reranker_model(features_filename=reranker_features, weights_filename=reranker_weights, **reranker_options)

    def parse(self, sentence):
        """
        Use BLLIP Parser to parse a sentence. Takes a sentence as a list
        of words; it will be automatically tagged with this BLLIP Parser
        instance's tagger.

        :return: An iterator that generates parse trees for the sentence
            from most likely to least likely.

        :param sentence: The sentence to be parsed
        :type sentence: list(str)
        :rtype: iter(Tree)
        """
        _ensure_ascii(sentence)
        nbest_list = self.rrp.parse(sentence)
        for scored_parse in nbest_list:
            yield _scored_parse_to_nltk_tree(scored_parse)

    def tagged_parse(self, word_and_tag_pairs):
        """
        Use BLLIP to parse a sentence. Takes a sentence as a list of
        (word, tag) tuples; the sentence must have already been tokenized
        and tagged. BLLIP will attempt to use the tags provided but may
        use others if it can't come up with a complete parse subject
        to those constraints. You may also specify a tag as ``None``
        to leave a token's tag unconstrained.

        :return: An iterator that generates parse trees for the sentence
            from most likely to least likely.

        :param sentence: Input sentence to parse as (word, tag) pairs
        :type sentence: list(tuple(str, str))
        :rtype: iter(Tree)
        """
        words = []
        tag_map = {}
        for i, (word, tag) in enumerate(word_and_tag_pairs):
            words.append(word)
            if tag is not None:
                tag_map[i] = tag
        _ensure_ascii(words)
        nbest_list = self.rrp.parse_tagged(words, tag_map)
        for scored_parse in nbest_list:
            yield _scored_parse_to_nltk_tree(scored_parse)

    @classmethod
    def from_unified_model_dir(cls, model_dir, parser_options=None, reranker_options=None):
        """
        Create a ``BllipParser`` object from a unified parsing model
        directory. Unified parsing model directories are a standardized
        way of storing BLLIP parser and reranker models together on disk.
        See ``bllipparser.RerankingParser.get_unified_model_parameters()``
        for more information about unified model directories.

        :return: A ``BllipParser`` object using the parser and reranker
            models in the model directory.

        :param model_dir: Path to the unified model directory.
        :type model_dir: str
        :param parser_options: optional dictionary of parser options, see
            ``bllipparser.RerankingParser.RerankingParser.load_parser_options()``
            for more information.
        :type parser_options: dict(str)
        :param reranker_options: optional dictionary of reranker options, see
            ``bllipparser.RerankingParser.RerankingParser.load_reranker_model()``
            for more information.
        :type reranker_options: dict(str)
        :rtype: BllipParser
        """
        parser_model_dir, reranker_features_filename, reranker_weights_filename = get_unified_model_parameters(model_dir)
        return cls(parser_model_dir, reranker_features_filename, reranker_weights_filename, parser_options, reranker_options)