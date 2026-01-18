class DistjunctUnion(DisjunctNode):
    """
    This node is true if *all* of its children are false. This is equivalent to
    ```
    disjunct(union(...))
    ```
    """
    JOINSTR = '|'