import pygame
import pygame_gui
import numpy as np
from collections import deque
from typing import List, Tuple, Deque, Dict, Any, Optional
import threading
import time
import random
import math
import asyncio
import os
import logging
import sys
import aiofiles
from functools import lru_cache as LRUCache
import aiohttp
import json
import cachetools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.utils.data.distributed as distributed
import torch.distributions as distributions
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.cuda as cuda  # Added for potential GPU acceleration
import torch.backends.cudnn as cudnn  # Added for optimizing deep learning computations on CUDA
import logging  # For detailed logging of operations and errors
import hashlib  # For generating unique identifiers for nodes
import bisect  # For maintaining sorted lists
import gc  # For explicit garbage collection if necessary
class GUI:

    def __init__(self, screen_dimensions: Tuple[int, int], font_size: int=20):
        """
        Initialize the GUI system for the Snake game, meticulously setting up the screen, font, and caching mechanisms to ensure optimal performance and user experience. This initialization now excludes the multiprocessing pool, focusing on a streamlined single-threaded operation while retaining the potential for future enhancements with parallel processing capabilities.

        Args:
            screen_dimensions (Tuple[int, int]): The width and height of the screen, specified in pixels, to define the area available for rendering the game's graphical content.
            font_size (int): The size of the font for text rendering, specified in points, which determines the visual clarity and readability of text displayed on the screen.

        Attributes:
            screen (pygame.Surface): The main screen surface for the game, which acts as the canvas where all graphical elements are drawn. This surface is initialized based on the provided screen dimensions.
            font (pygame.font.Font): Font used for rendering text, initialized with a default system font unless specified otherwise, and set to the provided font size to ensure text is legible.
            cache (cachetools.LRUCache): A least recently used (LRU) cache for storing pre-rendered text surfaces to improve rendering performance by avoiding redundant rendering operations for the same text content. The cache is set with a maximum size of 100 entries, balancing memory usage and performance.
        """
        pygame.init()
        self.screen = pygame.display.set_mode(screen_dimensions)
        self.font = pygame.font.Font(None, font_size)
        self.cache = cachetools.LRUCache(maxsize=100)
        self.manager = pygame_gui.UIManager(screen_dimensions, 'data/themes/theme.json')
        self.setup_elements()

    def setup_elements(self):
        """
        Set up the graphical elements of the game interface, including buttons, sliders, toggles, and other interactive components, to provide a visually appealing and user-friendly experience.

        Detailed Operations:
            - Create buttons for starting, pausing, restarting, and quitting the game.
            - Create sliders for adjusting the grid size and game speed.
            - Create toggles for enabling special game modes.
            - Create a dynamic scoreboard to display the current score.
            - Implement a background that changes color in a gradient pattern.
        """
        self.start_button = self._create_button('Start', (50, 50), self._start_game)
        self.pause_button = self._create_button('Pause', (150, 50), self._pause_game)
        self.restart_button = self._create_button('Restart', (250, 50), self._restart_game)
        self.quit_button = self._create_button('Quit', (350, 50), self._quit_game)
        self.grid_size_slider = self._create_slider('Grid Size', (50, 100), 10, 50, self._adjust_grid_size)
        self.game_speed_slider = self._create_slider('Game Speed', (50, 150), 1, 10, self._adjust_game_speed)
        self.special_mode_toggle = self._create_toggle('Special Mode', (50, 200), self._toggle_special_mode)
        self.scoreboard = self._create_scoreboard((50, 250), 'Score: 0')
        self._register_event_handlers()

    def _create_button(self, text, position, callback):
        button_rect = pygame.Rect(position, (100, 50))
        button = pygame_gui.elements.UIButton(relative_rect=button_rect, text=text, manager=self.manager)
        button.set_event_handler(callback)
        return button

    def _start_game(self):
        """
        Start the game, triggered by the start button.
        """
        print('Game started!')
        GameManager.snake.frozen = False

    def _create_scoreboard(self, position, initial_score):
        scoreboard_rect = pygame.Rect(position, (200, 100))
        scoreboard = pygame_gui.elements.UILabel(relative_rect=scoreboard_rect, text=str(initial_score), manager=self.manager)
        return scoreboard

    def _create_bubble(self, text, position):
        bubble_rect = pygame.Rect(position, (150, 100))
        bubble = pygame_gui.elements.UITooltip(relative_rect=bubble_rect, text=text, manager=self.manager)
        return bubble

    def _create_checkbox(self, text, position):
        checkbox_rect = pygame.Rect(position, (100, 50))
        checkbox = pygame_gui.elements.UICheckbox(relative_rect=checkbox_rect, text=text, manager=self.manager)
        return checkbox

    def _create_toggle(self, text, position, start_state):
        toggle_rect = pygame.Rect(position, (100, 50))
        toggle = pygame_gui.elements.UIButton(relative_rect=toggle_rect, text=text, manager=self.manager)
        toggle.set_event_handler(lambda: self._toggle_state(toggle, start_state))
        return toggle

    def _create_dropdown(self, options, position):
        dropdown_rect = pygame.Rect(position, (100, 50))
        dropdown = pygame_gui.elements.UIDropDownMenu(options=options, starting_option=options[0], relative_rect=dropdown_rect, manager=self.manager)
        return dropdown

    def _create_textbox(self, position, width):
        textbox_rect = pygame.Rect(position, (width, 50))
        textbox = pygame_gui.elements.UITextEntryLine(relative_rect=textbox_rect, manager=self.manager)
        return textbox

    def _create_slider(self, position, value_range, start_value):
        slider_rect = pygame.Rect(position, (100, 20))
        slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=slider_rect, start_value=start_value, value_range=value_range, manager=self.manager)
        return slider

    def _toggle_state(self, toggle, state):
        state = not state
        toggle.set_text('On' if state else 'Off')

    def _quit_game(self):
        """
        Quit the game, triggered by the quit button.
        """
        pygame.quit()
        sys.exit()

    async def draw_text(gui_instance, text: str, position: Tuple[int, int], color: Tuple[int, int, int]=(255, 255, 255)):
        """
        Asynchronously draw text on the screen at the specified position, utilizing caching to enhance performance significantly. This method handles the drawing operations in a non-blocking manner using asynchronous programming techniques, focusing on a single-threaded approach to ensure GUI responsiveness and performance even during intensive rendering operations.

        Args:
            gui_instance: The instance of the GUI class.
            text (str): The text to be rendered, which is a string of characters. This text is rendered onto the screen at the specified position and in the specified color.
            position (Tuple[int, int]): The (x, y) coordinates on the screen where the text will be drawn. The coordinates specify the top-left corner of the text surface.
            color (Tuple[int, int, int]): The RGB color of the text, specified as a tuple of three integers ranging from 0 to 255. This color determines how the text appears against the background.

        Detailed Operations:
            - Check if the text is already in the cache:
                - If not, render the text into a surface and store this surface in the cache.
                - If it is in the cache, retrieve the pre-rendered surface.
            - Perform the blitting (bit-block transfer) of the text surface onto the main screen surface at the specified position.
            - This operation is performed asynchronously to ensure that the GUI remains responsive and performant even during intensive rendering operations.
        """
        if text not in gui_instance.cache:
            text_surface = gui_instance.font.render(text, True, color)
            gui_instance.cache[text] = text_surface
        else:
            text_surface = gui_instance.cache[text]
        await asyncio.to_thread(gui_instance.screen.blit, text_surface, position)

    def update_display(self):
        """
        Update the display to reflect the latest changes, ensuring all drawn elements are visible. This method is crucial for maintaining the visual integrity of the game's interface after any graphical updates.

        Detailed Operations:
            - Flip the display: This operation updates the actual display seen by the user with all the changes made to the screen surface during the current frame. It is equivalent to refreshing the screen to show the latest graphical content.
        """
        pygame.display.flip()

    def clear_screen(self, color: Tuple[int, int, int]=(0, 0, 0)):
        """
        Clear the screen with a uniform color to prepare for the next frame, ensuring a clean visual state before any new drawing operations are performed.

        Args:
            color (Tuple[int, int, int]): The RGB color to fill the screen, specified as a tuple of three integers ranging from 0 to 255. This color becomes the background color of the screen until the next clearing operation.

        Detailed Operations:
            - Fill the screen: The entire screen surface is filled with the specified color, effectively clearing any previous graphical content and setting a uniform background color for new drawings.
        """
        self.screen.fill(color)