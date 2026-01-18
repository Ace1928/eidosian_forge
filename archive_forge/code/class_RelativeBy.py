from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
class RelativeBy:
    """Gives the opportunity to find elements based on their relative location
    on the page from a root elelemt. It is recommended that you use the helper
    function to create it.

    Example:
        lowest = driver.find_element(By.ID, "below")

        elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").above(lowest))

        ids = [el.get_attribute('id') for el in elements]
        assert "above" in ids
        assert "mid" in ids
    """

    def __init__(self, root: Optional[Dict[Union[By, str], str]]=None, filters: Optional[List]=None):
        """Creates a new RelativeBy object. It is preferred if you use the
        `locate_with` method as this signature could change.

        :Args:
            root - A dict with `By` enum as the key and the search query as the value
            filters - A list of the filters that will be searched. If none are passed
                in please use the fluent API on the object to create the filters
        """
        self.root = root
        self.filters = filters or []

    def above(self, element_or_locator: Union[WebElement, Dict]=None) -> 'RelativeBy':
        """Add a filter to look for elements above.

        :Args:
            - element_or_locator: Element to look above
        """
        if not element_or_locator:
            raise WebDriverException('Element or locator must be given when calling above method')
        self.filters.append({'kind': 'above', 'args': [element_or_locator]})
        return self

    def below(self, element_or_locator: Union[WebElement, Dict]=None) -> 'RelativeBy':
        """Add a filter to look for elements below.

        :Args:
            - element_or_locator: Element to look below
        """
        if not element_or_locator:
            raise WebDriverException('Element or locator must be given when calling below method')
        self.filters.append({'kind': 'below', 'args': [element_or_locator]})
        return self

    def to_left_of(self, element_or_locator: Union[WebElement, Dict]=None) -> 'RelativeBy':
        """Add a filter to look for elements to the left of.

        :Args:
            - element_or_locator: Element to look to the left of
        """
        if not element_or_locator:
            raise WebDriverException('Element or locator must be given when calling to_left_of method')
        self.filters.append({'kind': 'left', 'args': [element_or_locator]})
        return self

    def to_right_of(self, element_or_locator: Union[WebElement, Dict]=None) -> 'RelativeBy':
        """Add a filter to look for elements right of.

        :Args:
            - element_or_locator: Element to look right of
        """
        if not element_or_locator:
            raise WebDriverException('Element or locator must be given when calling to_right_of method')
        self.filters.append({'kind': 'right', 'args': [element_or_locator]})
        return self

    def near(self, element_or_locator_distance: Union[WebElement, Dict, int]=None) -> 'RelativeBy':
        """Add a filter to look for elements near.

        :Args:
            - element_or_locator_distance: Element to look near by the element or within a distance
        """
        if not element_or_locator_distance:
            raise WebDriverException('Element or locator or distance must be given when calling near method')
        self.filters.append({'kind': 'near', 'args': [element_or_locator_distance]})
        return self

    def to_dict(self) -> Dict:
        """Create a dict that will be passed to the driver to start searching
        for the element."""
        return {'relative': {'root': self.root, 'filters': self.filters}}