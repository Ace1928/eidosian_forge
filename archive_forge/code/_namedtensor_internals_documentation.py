from collections import OrderedDict
There are two usages:

    tensor.rename(*names) returns a view on tensor with named dims `names`.
    `names` must be of length `tensor.dim()`; otherwise, if '...' is in `names`,
    then it is expanded greedily to be equal to the corresponding names from
    `tensor.names`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename('...', 'height', 'width').names
    ('N', 'C', 'height', 'width')

    >>> # xdoctest: +SKIP
    >>> x.rename('batch', '...', 'width').names
    ('batch', 'C', 'H', 'width')

    ```

    tensor.rename(**rename_map) returns a view on tensor that has rename dims
        as specified in the mapping `rename_map`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename(W='width', H='height').names
    ('N', 'C', 'height', 'width')

    ```

    Finally, tensor.rename has an in-place version called tensor.rename_.
    