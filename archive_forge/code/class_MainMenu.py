import pygame
from Constants import *
from GA import *
import sys
class MainMenu(Menu):

    def __init__(self, game):
        Menu.__init__(self, game)
        self.state = 'BFS'
        self.cursorBFS = MENU_COLOR
        self.cursorDFS = WHITE
        self.cursorASTAR = WHITE
        self.cursorGA = WHITE
        self.BFSx, self.BFSy = (self.mid_size, self.mid_size - 50)
        self.DFSx, self.DFSy = (self.mid_size, self.mid_size + 0)
        self.ASTARx, self.ASTARy = (self.mid_size, self.mid_size + 50)
        self.GAx, self.GAy = (self.mid_size, self.mid_size + 100)
        self.cursor_rect.midtop = (self.BFSx + self.offset, self.BFSy)

    def change_cursor_color(self):
        self.clear_cursor_color()
        if self.state == 'BFS':
            self.cursorBFS = MENU_COLOR
        elif self.state == 'DFS':
            self.cursorDFS = MENU_COLOR
        elif self.state == 'ASTAR':
            self.cursorASTAR = MENU_COLOR
        elif self.state == 'GA':
            self.cursorGA = MENU_COLOR

    def clear_cursor_color(self):
        self.cursorBFS = WHITE
        self.cursorDFS = WHITE
        self.cursorASTAR = WHITE
        self.cursorGA = WHITE

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.event_handler()
            self.check_input()
            self.game.display.fill(WINDOW_COLOR)
            self.game.draw_text('Ai Snake Game', size=self.title_size, x=self.game.SIZE / 2, y=self.game.SIZE / 2 - 2 * (CELL_SIZE + NO_OF_CELLS), color=TITLE_COLOR)
            self.game.draw_text('BFS', size=self.option_size, x=self.BFSx, y=self.BFSy, color=self.cursorBFS)
            self.game.draw_text('DFS', size=self.option_size, x=self.DFSx, y=self.DFSy, color=self.cursorDFS)
            self.game.draw_text('AStar', size=self.option_size, x=self.ASTARx, y=self.ASTARy, color=self.cursorASTAR)
            self.game.draw_text('Genetic Algorithm', size=self.option_size, x=self.GAx, y=self.GAy, color=self.cursorGA)
            self.draw_cursor()
            self.change_cursor_color()
            self.blit_menu()

    def check_input(self):
        self.move_cursor()
        if self.game.START:
            if self.state == 'GA':
                self.game.curr_menu = self.game.GA
            else:
                self.game.playing = True
            self.run_display = False

    def move_cursor(self):
        if self.game.DOWNKEY:
            if self.state == 'BFS':
                self.cursor_rect.midtop = (self.DFSx + self.offset, self.DFSy)
                self.state = 'DFS'
            elif self.state == 'DFS':
                self.cursor_rect.midtop = (self.ASTARx + self.offset, self.ASTARy)
                self.state = 'ASTAR'
            elif self.state == 'ASTAR':
                self.cursor_rect.midtop = (self.GAx + self.offset, self.GAy)
                self.state = 'GA'
            elif self.state == 'GA':
                self.cursor_rect.midtop = (self.BFSx + self.offset, self.BFSy)
                self.state = 'BFS'
        if self.game.UPKEY:
            if self.state == 'BFS':
                self.cursor_rect.midtop = (self.GAx + self.offset, self.GAy)
                self.state = 'GA'
            elif self.state == 'DFS':
                self.cursor_rect.midtop = (self.BFSx + self.offset, self.BFSy)
                self.state = 'BFS'
            elif self.state == 'ASTAR':
                self.cursor_rect.midtop = (self.DFSx + self.offset, self.DFSy)
                self.state = 'DFS'
            elif self.state == 'GA':
                self.cursor_rect.midtop = (self.ASTARx + self.offset, self.ASTARy)
                self.state = 'ASTAR'