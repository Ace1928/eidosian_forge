import itertools
import time
from typing import Dict, List, Optional, Union
from redis.client import Pipeline
from redis.utils import deprecated_function
from ..helpers import get_protocol_version, parse_to_dict
from ._util import to_string
from .aggregation import AggregateRequest, AggregateResult, Cursor
from .document import Document
from .query import Query
from .result import Result
from .suggestion import SuggestionParser
class AsyncSearchCommands(SearchCommands):

    async def info(self):
        """
        Get info an stats about the the current index, including the number of
        documents, memory consumption, etc

        For more information see `FT.INFO <https://redis.io/commands/ft.info>`_.
        """
        res = await self.execute_command(INFO_CMD, self.index_name)
        return self._parse_results(INFO_CMD, res)

    async def search(self, query: Union[str, Query], query_params: Dict[str, Union[str, int, float]]=None):
        """
        Search the index for a given query, and return a result of documents

        ### Parameters

        - **query**: the search query. Either a text for simple queries with
                     default parameters, or a Query object for complex queries.
                     See RediSearch's documentation on query format

        For more information see `FT.SEARCH <https://redis.io/commands/ft.search>`_.
        """
        args, query = self._mk_query_args(query, query_params=query_params)
        st = time.time()
        res = await self.execute_command(SEARCH_CMD, *args)
        if isinstance(res, Pipeline):
            return res
        return self._parse_results(SEARCH_CMD, res, query=query, duration=(time.time() - st) * 1000.0)

    async def aggregate(self, query: Union[str, Query], query_params: Dict[str, Union[str, int, float]]=None):
        """
        Issue an aggregation query.

        ### Parameters

        **query**: This can be either an `AggregateRequest`, or a `Cursor`

        An `AggregateResult` object is returned. You can access the rows from
        its `rows` property, which will always yield the rows of the result.

        For more information see `FT.AGGREGATE <https://redis.io/commands/ft.aggregate>`_.
        """
        if isinstance(query, AggregateRequest):
            has_cursor = bool(query._cursor)
            cmd = [AGGREGATE_CMD, self.index_name] + query.build_args()
        elif isinstance(query, Cursor):
            has_cursor = True
            cmd = [CURSOR_CMD, 'READ', self.index_name] + query.build_args()
        else:
            raise ValueError('Bad query', query)
        cmd += self.get_params_args(query_params)
        raw = await self.execute_command(*cmd)
        return self._parse_results(AGGREGATE_CMD, raw, query=query, has_cursor=has_cursor)

    async def spellcheck(self, query, distance=None, include=None, exclude=None):
        """
        Issue a spellcheck query

        ### Parameters

        **query**: search query.
        **distance***: the maximal Levenshtein distance for spelling
                       suggestions (default: 1, max: 4).
        **include**: specifies an inclusion custom dictionary.
        **exclude**: specifies an exclusion custom dictionary.

        For more information see `FT.SPELLCHECK <https://redis.io/commands/ft.spellcheck>`_.
        """
        cmd = [SPELLCHECK_CMD, self.index_name, query]
        if distance:
            cmd.extend(['DISTANCE', distance])
        if include:
            cmd.extend(['TERMS', 'INCLUDE', include])
        if exclude:
            cmd.extend(['TERMS', 'EXCLUDE', exclude])
        res = await self.execute_command(*cmd)
        return self._parse_results(SPELLCHECK_CMD, res)

    async def config_set(self, option: str, value: str) -> bool:
        """Set runtime configuration option.

        ### Parameters

        - **option**: the name of the configuration option.
        - **value**: a value for the configuration option.

        For more information see `FT.CONFIG SET <https://redis.io/commands/ft.config-set>`_.
        """
        cmd = [CONFIG_CMD, 'SET', option, value]
        raw = await self.execute_command(*cmd)
        return raw == 'OK'

    async def config_get(self, option: str) -> str:
        """Get runtime configuration option value.

        ### Parameters

        - **option**: the name of the configuration option.

        For more information see `FT.CONFIG GET <https://redis.io/commands/ft.config-get>`_.
        """
        cmd = [CONFIG_CMD, 'GET', option]
        res = {}
        res = await self.execute_command(*cmd)
        return self._parse_results(CONFIG_CMD, res)

    async def load_document(self, id):
        """
        Load a single document by id
        """
        fields = await self.client.hgetall(id)
        f2 = {to_string(k): to_string(v) for k, v in fields.items()}
        fields = f2
        try:
            del fields['id']
        except KeyError:
            pass
        return Document(id=id, **fields)

    async def sugadd(self, key, *suggestions, **kwargs):
        """
        Add suggestion terms to the AutoCompleter engine. Each suggestion has
        a score and string.
        If kwargs["increment"] is true and the terms are already in the
        server's dictionary, we increment their scores.

        For more information see `FT.SUGADD <https://redis.io/commands/ft.sugadd>`_.
        """
        pipe = self.pipeline(transaction=False)
        for sug in suggestions:
            args = [SUGADD_COMMAND, key, sug.string, sug.score]
            if kwargs.get('increment'):
                args.append('INCR')
            if sug.payload:
                args.append('PAYLOAD')
                args.append(sug.payload)
            pipe.execute_command(*args)
        return (await pipe.execute())[-1]

    async def sugget(self, key: str, prefix: str, fuzzy: bool=False, num: int=10, with_scores: bool=False, with_payloads: bool=False) -> List[SuggestionParser]:
        """
        Get a list of suggestions from the AutoCompleter, for a given prefix.

        Parameters:

        prefix : str
            The prefix we are searching. **Must be valid ascii or utf-8**
        fuzzy : bool
            If set to true, the prefix search is done in fuzzy mode.
            **NOTE**: Running fuzzy searches on short (<3 letters) prefixes
            can be very
            slow, and even scan the entire index.
        with_scores : bool
            If set to true, we also return the (refactored) score of
            each suggestion.
            This is normally not needed, and is NOT the original score
            inserted into the index.
        with_payloads : bool
            Return suggestion payloads
        num : int
            The maximum number of results we return. Note that we might
            return less. The algorithm trims irrelevant suggestions.

        Returns:

        list:
             A list of Suggestion objects. If with_scores was False, the
             score of all suggestions is 1.

        For more information see `FT.SUGGET <https://redis.io/commands/ft.sugget>`_.
        """
        args = [SUGGET_COMMAND, key, prefix, 'MAX', num]
        if fuzzy:
            args.append(FUZZY)
        if with_scores:
            args.append(WITHSCORES)
        if with_payloads:
            args.append(WITHPAYLOADS)
        ret = await self.execute_command(*args)
        results = []
        if not ret:
            return results
        parser = SuggestionParser(with_scores, with_payloads, ret)
        return [s for s in parser]