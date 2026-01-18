import pygame
import sys
import random
import numpy as np
import duckdb
from multiprocessing import Pool
import functools

pygame.init()
size = width, height = 800, 600
screen = pygame.display.set_mode(size)
center = np.array([width / 2, height / 2])

# Constants
GRAVITY_STRENGTH = 0.0005
CHARGE_CONSTANT = 0.00001
QUANTUM_FORCE_CONSTANT = 0.0001
NUCLEAR_FORCE_RANGE = 100
NUCLEAR_FORCE_STRENGTH = 0.002
WEAK_FORCE_STRENGTH = 0.001
DARK_MATTER_PULL = 0.0003


class Particle:
    def __init__(self, x, y, vx, vy, mass, charge, type):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.mass = mass
        self.charge = charge
        self.type = type

    def apply_gravity(self):
        direction_to_center = center - self.pos
        gravity_force = direction_to_center * (
            GRAVITY_STRENGTH * self.mass / np.linalg.norm(direction_to_center) ** 3
        )
        self.vel += gravity_force

    def apply_dark_matter_pull(self):
        direction_to_center = center - self.pos
        pull_force = direction_to_center * DARK_MATTER_PULL
        self.vel += pull_force

    def apply_electromagnetic_force(self, other):
        displacement = other.pos - self.pos
        distance = np.linalg.norm(displacement)
        force_magnitude = CHARGE_CONSTANT * self.charge * other.charge / (distance**2)
        force_direction = displacement / distance
        self.vel += force_direction * force_magnitude / self.mass

    def quantum_fluctuations(self):
        self.vel += np.random.randn(2) * QUANTUM_FORCE_CONSTANT

    def nuclear_interactions(self, particles):
        for other in particles:
            if (
                self is not other
                and np.linalg.norm(other.pos - self.pos) < NUCLEAR_FORCE_RANGE
            ):
                if self.type == "hydrogen" and other.type == "hydrogen":
                    self.fuse(other, particles)

    def fuse(self, other, particles):
        if random.random() < 0.05:  # Simplified probability of fusion
            new_mass = self.mass + other.mass
            new_pos = (self.pos + other.pos) / 2
            new_vel = (self.vel + other.vel) / 2
            helium = Particle(
                new_pos[0], new_pos[1], new_vel[0], new_vel[1], new_mass, 0, "helium"
            )
            particles.append(helium)
            particles.remove(self)
            particles.remove(other)

    def update(self, particles):
        self.apply_gravity()
        self.apply_dark_matter_pull()
        for other in particles:
            self.apply_electromagnetic_force(other)
        self.quantum_fluctuations()
        self.nuclear_interactions(particles)
        self.pos += self.vel
        self.pos = np.clip(self.pos, 0, np.array([width, height]))

    def draw(self):
        color = (255, 0, 0) if self.type == "hydrogen" else (0, 0, 255)
        pygame.draw.circle(
            screen, color, self.pos.astype(int), 1
        )  # Smaller particle size


def initialize_particles():
    particles = []
    for _ in range(1000):  # Increased number of particles
        x, y = random.randrange(width), random.randrange(height)
        vx, vy = random.uniform(-1, 1), random.uniform(-1, 1)
        mass = 1
        charge = random.choice([-1, 1])
        type = "hydrogen"
        particles.append(Particle(x, y, vx, vy, mass, charge, type))
    return particles


def update_particles(particles):
    with Pool(processes=4) as pool:
        particles = pool.map(
            functools.partial(update_particle, particles=particles), particles
        )
    return particles


def run():
    particles = initialize_particles()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))  # Clear the screen
        particles = update_particles(particles)
        for particle in particles:
            particle.draw()
        pygame.display.flip()
        clock.tick(60)  # Maintain 60 frames per second

    pygame.quit()
    sys.exit()


run()
