import pygame
from Constants import *
from GA import *
import sys
class TextBox:

    def __init__(self, x, y, game):
        self.font = FONT
        self.input_rect = pygame.Rect(x, y, TXT_WIDTH, TXT_HEIGHT)
        self.input = ''
        self.game = game
        self.active = False

    def draw_input(self):
        pos = pygame.mouse.get_pos()
        if self.input_rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                self.active = True
        elif pygame.mouse.get_pressed()[0] == 1:
            self.active = False
        if self.active:
            color = TXT_ACTIVE
        else:
            color = TXT_PASSIVE
        pygame.draw.rect(self.game.display, color, self.input_rect, 2)
        text_surface = self.font.render(self.input, False, WHITE)
        self.game.display.blit(text_surface, (self.input_rect.x + 15, self.input_rect.y + 1))