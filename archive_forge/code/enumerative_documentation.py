important invariant is that the parts on the stack are themselves in
Returns the number of partitions of a multiset whose components
        have the multiplicities given in ``multiplicities``.

        For larger counts, this method is much faster than calling one
        of the enumerators and counting the result.  Uses dynamic
        programming to cut down on the number of nodes actually
        explored.  The dictionary used in order to accelerate the
        counting process is stored in the ``MultisetPartitionTraverser``
        object and persists across calls.  If the user does not
        expect to call ``count_partitions`` for any additional
        multisets, the object should be cleared to save memory.  On
        the other hand, the cache built up from one count run can
        significantly speed up subsequent calls to ``count_partitions``,
        so it may be advantageous not to clear the object.

        Examples
        ========

        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> m.count_partitions([9,8,2])
        288716
        >>> m.count_partitions([2,2])
        9
        >>> del m

        Notes
        =====

        If one looks at the workings of Knuth's algorithm M [AOCP]_, it
        can be viewed as a traversal of a binary tree of parts.  A
        part has (up to) two children, the left child resulting from
        the spread operation, and the right child from the decrement
        operation.  The ordinary enumeration of multiset partitions is
        an in-order traversal of this tree, and with the partitions
        corresponding to paths from the root to the leaves. The
        mapping from paths to partitions is a little complicated,
        since the partition would contain only those parts which are
        leaves or the parents of a spread link, not those which are
        parents of a decrement link.

        For counting purposes, it is sufficient to count leaves, and
        this can be done with a recursive in-order traversal.  The
        number of leaves of a subtree rooted at a particular part is a
        function only of that part itself, so memoizing has the
        potential to speed up the counting dramatically.

        This method follows a computational approach which is similar
        to the hypothetical memoized recursive function, but with two
        differences:

        1) This method is iterative, borrowing its structure from the
           other enumerations and maintaining an explicit stack of
           parts which are in the process of being counted.  (There
           may be multisets which can be counted reasonably quickly by
           this implementation, but which would overflow the default
           Python recursion limit with a recursive implementation.)

        2) Instead of using the part data structure directly, a more
           compact key is constructed.  This saves space, but more
           importantly coalesces some parts which would remain
           separate with physical keys.

        Unlike the enumeration functions, there is currently no _range
        version of count_partitions.  If someone wants to stretch
        their brain, it should be possible to construct one by
        memoizing with a histogram of counts rather than a single
        count, and combining the histograms.
        