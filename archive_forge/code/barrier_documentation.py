from typing import Any, Callable, List, Optional, Tuple
Reset barrier, removing all received results.

        Resetting the barrier will reset the completion status. When ``max_results``
        is set and enough new events arrive after resetting, the
        :meth:`on_completion` callback will be invoked again.
        