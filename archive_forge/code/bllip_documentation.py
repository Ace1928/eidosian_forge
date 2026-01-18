from nltk.parse.api import ParserI
from nltk.tree import Tree
from NLTK's downloader. More unified parsing models can be obtained with

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
        