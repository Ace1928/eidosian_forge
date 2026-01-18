from typing import List
from typing import Optional
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.nodes import Item
def _should_truncate_item(item: Item) -> bool:
    """Whether or not this test item is eligible for truncation."""
    verbose = item.config.get_verbosity(Config.VERBOSITY_ASSERTIONS)
    return verbose < 2 and (not util.running_on_ci())