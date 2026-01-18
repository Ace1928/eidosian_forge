from abc import ABC, abstractmethod
from typing import List, Optional
class DisjunctiveTrie:

    def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
        """
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """
        self.max_height = max([len(one) for one in nested_token_ids])
        root = {}
        for token_ids in nested_token_ids:
            level = root
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}
                level = level[token_id]
        if no_subsets and self.has_subsets(root, nested_token_ids):
            raise ValueError(f"Each list in `nested_token_ids` can't be a complete subset of another list, but is {nested_token_ids}.")
        self.trie = root

    def next_tokens(self, current_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.trie
        for current_token in current_seq:
            start = start[current_token]
        next_tokens = list(start.keys())
        return next_tokens

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)
        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])

    def has_subsets(self, trie, nested_token_ids):
        """
        Returns whether # of leaves == # of words. Otherwise some word is a subset of another.
        """
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count