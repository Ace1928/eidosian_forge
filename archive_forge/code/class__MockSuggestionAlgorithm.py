from typing import Dict, List, Optional
from ray.tune.search import Searcher, ConcurrencyLimiter
from ray.tune.search.search_generator import SearchGenerator
from ray.tune.experiment import Trial
class _MockSuggestionAlgorithm(SearchGenerator):

    def __init__(self, max_concurrent: Optional[int]=None, **kwargs):
        self.searcher = _MockSearcher(**kwargs)
        if max_concurrent:
            self.searcher = ConcurrencyLimiter(self.searcher, max_concurrent=max_concurrent)
        super(_MockSuggestionAlgorithm, self).__init__(self.searcher)

    @property
    def live_trials(self) -> List[Trial]:
        return self.searcher.live_trials

    @property
    def results(self) -> List[Dict]:
        return self.searcher.results