import enum
import sys
class TraversalStrategy(enum.Enum, **strictEnum):
    BreadthFirstSearch = 1
    PrefixDepthFirstSearch = 2
    PostfixDepthFirstSearch = 3
    BFS = BreadthFirstSearch
    ParentLastDepthFirstSearch = PostfixDepthFirstSearch
    PostfixDFS = PostfixDepthFirstSearch
    ParentFirstDepthFirstSearch = PrefixDepthFirstSearch
    PrefixDFS = PrefixDepthFirstSearch
    DepthFirstSearch = PrefixDepthFirstSearch
    DFS = DepthFirstSearch