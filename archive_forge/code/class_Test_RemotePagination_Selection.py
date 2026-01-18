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
class Test_RemotePagination_Selection(Test_RemotePagination):
    selectable = True

    def test_one_item_first_page(self, page):
        rows = self.get_rows(page)
        rows.nth(0).click()
        self.check_selected(page, [0])
        with self.hold_down_ctrl(page):
            rows.nth(0).click()
        self.check_selected(page, [])

    def test_one_item_first_page_and_then_another(self, page):
        rows = self.get_rows(page)
        rows.nth(0).click()
        self.check_selected(page, [0])
        rows.nth(1).click()
        self.check_selected(page, [1])

    def test_two_items_first_page(self, page):
        rows = self.get_rows(page)
        rows.nth(0).click()
        self.check_selected(page, [0])
        with self.hold_down_ctrl(page):
            rows.nth(1).click()
        self.check_selected(page, [0, 1])

    def test_one_item_first_page_goto_second_page(self, page):
        rows = self.get_rows(page)
        rows.nth(0).click()
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

    def test_one_item_both_pages(self, page):
        rows = self.get_rows(page)
        rows.nth(0).click()
        self.check_selected(page, [0], 1)
        self.goto_page(page, 2)
        rows = self.get_rows(page)
        with self.hold_down_ctrl(page):
            rows.nth(0).click()
        self.check_selected(page, [0, 10], 1)

    def test_one_item_and_then_second_page(self, page):
        rows = self.get_rows(page)
        rows.nth(0).click()
        self.check_selected(page, [0], 1)
        self.goto_page(page, 2)
        rows = self.get_rows(page)
        rows.nth(0).click()
        self.check_selected(page, [10], 1)

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

    @pytest.mark.parametrize('selection', (0, 10), ids=['page1', 'page2'])
    def test_filtering(self, page, selection):
        self.widget.selection = [selection]
        self.check_selected(page, [selection], int(selection == 0))
        self.set_filtering(page, selection)
        self.check_selected(page, [selection], 1)
        self.set_filtering(page, 1)
        self.check_selected(page, [selection], 0)

    def test_shift_select_page_1(self, page):
        rows = self.get_rows(page)
        with self.hold_down_shift(page):
            rows.nth(0).click()
            rows.nth(2).click()
        self.check_selected(page, [0, 1, 2])
        self.goto_page(page, 2)
        self.check_selected(page, [0, 1, 2], 0)
        self.goto_page(page, 1)
        self.check_selected(page, [0, 1, 2])

    def test_shift_select_page_2(self, page):
        self.check_selected(page, [])
        self.goto_page(page, 2)
        rows = self.get_rows(page)
        with self.hold_down_shift(page):
            rows.nth(0).click()
            rows.nth(2).click()
        self.check_selected(page, [10, 11, 12])
        self.goto_page(page, 1)
        self.check_selected(page, [10, 11, 12], 0)

    def test_shift_select_both_pages(self, page):
        rows = self.get_rows(page)
        with self.hold_down_shift(page):
            rows.nth(0).click()
            rows.nth(2).click()
        self.check_selected(page, [0, 1, 2])
        self.goto_page(page, 2)
        rows = self.get_rows(page)
        with self.hold_down_shift(page):
            rows.nth(0).click()
            rows.nth(2).click()
        self.check_selected(page, [0, 1, 2, 10, 11, 12], 3)
        self.goto_page(page, 1)
        self.check_selected(page, [0, 1, 2, 10, 11, 12], 3)