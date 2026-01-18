from bs4 import BeautifulSoup
def _get_dash_dom_by_attribute(self, attr):
    return BeautifulSoup(self.find_element(self.dash_entry_locator).get_attribute(attr), 'lxml')