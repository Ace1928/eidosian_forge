from typing import Any, Generic, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar
Removes and returns an item from the priority queue.

        Returns:
            A tuple whose first element is the priority of the dequeued item
            and whose second element is the dequeued item.

        Raises:
            ValueError:
                The queue is empty.
        