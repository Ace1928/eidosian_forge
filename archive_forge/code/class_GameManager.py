import pygame
import random
import heapq
import logging
from typing import List, Optional, Dict, Any, Tuple
import cProfile
from collections import deque
import numpy as np
import time
import torch
from functools import lru_cache as LRUCache
import math
import asyncio
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue
from collections import defaultdict
class GameManager:

    def __init__(self, snake=Snake(), grid=Grid(WIDTH, HEIGHT), food=Food(None)):
        self.snake = snake
        self.grid = grid
        self.food = food
        self.decision_maker = DecisionMaker(self.snake, self.grid, self.food)
        self.pathfinderTheta = ThetaStar(self.grid)
        self.pathfinderCDP = ConstrainedDelaunayPathfinder(Grid.get_points(self.grid), Grid.get_obstacles(self.grid))
        self.pathfinderAHP = AmoebaHamiltonianPathfinder(self.snake, self.grid, self.food)
        self.score = 0
        self.high_score = 0
        self.moves = 0
        self.game_over = False
        self.game_won = False
        self.game_speed = 10
        self.FPS = 60
        self.WIN = pygame.display.set_mode((WIDTH * TILE_SIZE, HEIGHT * TILE_SIZE))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_color = (255, 255, 255)
        self.running = True

    def reset(self):
        self.snake = Snake()
        self.grid = Grid(WIDTH, HEIGHT)
        self.food = Food(None)
        self.decision_maker = DecisionMaker(self.snake, self.grid, self.food)
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.game_won = False
        self.game_speed = 10
        self.FPS = 60
        self.WIN = pygame.display.set_mode((WIDTH * TILE_SIZE, HEIGHT * TILE_SIZE))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_color = (255, 255, 255)
        self.running = True

    def update(self):
        self.snake.update()
        if self.snake.alive:
            if self.snake.get_head_position() == self.food.position:
                self.snake.length += 1
                self.score += 1
                self.food.randomize_position(self.snake.segments)
                if self.score > self.high_score:
                    self.high_score = self.score
            self.moves += 1
        else:
            self.game_over = True
            self.running = False

    def draw(self):
        self.WIN.fill((0, 0, 0))
        self.snake.draw(self.WIN)
        self.food.draw(self.WIN)
        self.draw_score()
        pygame.display.update()

    def draw_score(self):
        score_text = self.font.render(f'Score: {self.score}', True, self.font_color)
        high_score_text = self.font.render(f'High Score: {self.high_score}', True, self.font_color)
        self.WIN.blit(score_text, (10, 10))
        self.WIN.blit(high_score_text, (10, 40))

    def run(self):
        while self.running:
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            if not self.game_over:
                if self.moves % self.game_speed == 0:
                    next_move = asyncio.run(self.decision_maker.decide_next_move())
                    self.snake.change_direction(next_move)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.running = False
                            if event.key == pygame.K_UP:
                                self.snake.change_direction('up')
                            if event.key == pygame.K_DOWN:
                                self.snake.change_direction('down')
                            if event.key == pygame.K_LEFT:
                                self.snake.change_direction('left')
                            if event.key == pygame.K_RIGHT:
                                self.snake.change_direction('right')
                            if event.key == pygame.K_Q:
                                self.game_speed -= 10
                            if event.key == pygame.K_E:
                                self.game_speed += 10
                            if event.key == pygame.K_R:
                                self.reset()
                    self.update()
                self.draw()
            else:
                self.reset()
        pygame.quit()