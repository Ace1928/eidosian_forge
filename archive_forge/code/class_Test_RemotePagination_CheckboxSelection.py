from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import numpy as np
import pandas as pd
import param
import pytest
from bokeh.models.widgets.tables import (
from playwright.sync_api import expect
from panel.depends import bind
from panel.io.state import state
from panel.layout.base import Column
from panel.models.tabulator import _TABULATOR_THEMES_MAPPING
from panel.tests.util import get_ctrl_modifier, serve_component, wait_until
from panel.widgets import Select, Tabulator
class Test_RemotePagination_CheckboxSelection(Test_RemotePagination):
    selectable = 'checkbox'

    def get_checkboxes(self, page):
        return page.locator('input[type="checkbox"]')

    def test_full_firstpage(self, page):
        checkboxes = self.get_checkboxes(page)
        checkboxes.nth(0).click()
        self.check_selected(page, list(range(10)))
        checkboxes.last.click()
        self.check_selected(page, list(range(9)))

    def test_one_item_first_page(self, page):
        checkboxes = self.get_checkboxes(page)
        checkboxes.nth(1).click()
        self.check_selected(page, [0])
        checkboxes.nth(1).click()
        self.check_selected(page, [])

    def test_one_item_first_page_goto_second_page(self, page):
        checkboxes = self.get_checkboxes(page)
        checkboxes.nth(1).click()
        self.check_selected(page, [0], 1)
        self.goto_page(page, 2)
        self.check_selected(page, [0], 0)
        self.goto_page(page, 1)
        self.check_selected(page, [0], 1)

    def test_one_item_both_pages_python(self, page):
        self.widget.selection = [0, 10]
        self.check_selected(page, [0, 10], 1)
        self.goto_page(page, 2)
        self.check_selected(page, [0, 10], 1)

    @pytest.mark.parametrize('selection', (0, 10), ids=['page1', 'page2'])
    def test_sorting(self, page, selection):
        self.widget.selection = [selection]
        self.check_selected(page, [selection], int(selection == 0))
        self.click_sorting(page)
        self.check_selected(page, [selection], int(selection == 0))
        self.click_sorting(page)
        self.check_selected(page, [selection], int(selection == 10))
        self.click_sorting(page)
        self.check_selected(page, [selection], int(selection == 0))

    def test_sorting_all(self, page):
        checkboxes = self.get_checkboxes(page)
        checkboxes.nth(0).click()
        self.click_sorting(page)
        self.check_selected(page, list(range(10)), 10)
        self.click_sorting(page)
        self.check_selected(page, list(range(10)), 0)
        self.click_sorting(page)
        self.check_selected(page, list(range(10)), 10)

    @pytest.mark.parametrize('selection', (0, 10), ids=['page1', 'page2'])
    def test_filtering(self, page, selection):
        self.widget.selection = [selection]
        self.check_selected(page, [selection], int(selection == 0))
        self.set_filtering(page, selection)
        self.check_selected(page, [selection], 1)
        self.set_filtering(page, 1)
        self.check_selected(page, [selection], 0)

    def test_filtering_all(self, page):
        checkboxes = self.get_checkboxes(page)
        checkboxes.nth(0).click()
        for n in range(10):
            self.set_filtering(page, n)
            self.check_selected(page, list(range(10)), 1)
        for n in range(10, 20):
            self.set_filtering(page, n)
            self.check_selected(page, list(range(10)), 0)
            expect(page.locator('.tabulator')).to_have_count(1)