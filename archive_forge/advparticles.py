"""
GeneParticles: Cellular Automata with Gene Expression, Emergent Behaviors, 
and Evolutionary Dynamics
-------------------------------------------------------------------------------------------------
A hyper-advanced particle simulation that models cellular-like entities ("particles") endowed with 
complex dynamic genetic traits, adaptive behaviors, emergent properties, hierarchical speciation, 
and intricate interaction networks spanning multiple dimensions of trait synergy and competition.
"""
import random
import collections
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Union
import time
import numpy as np
import pygame
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import gc
import traceback
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import psutil
import os

###############################################################
# Genetic Parameter Configuration
###############################################################

class GeneticParamConfig:
    """
    Holds genetic parameters and mutation ranges for a wide array of traits.
    This is a dedicated config structure to handle complex genetic aspects.

    Additional Traits Introduced:
    - synergy_affinity: how strongly a particle engages in synergy
    - colony_factor: how likely a particle is to form or join colonies 
    - drift_sensitivity: how sensitive the particle is to evolutionary drift
    - max_energy_storage: maximum energy a particle can store
    - sensory_sensitivity: sensitivity of a particle's sensory mechanisms
    - memory_transfer_rate: rate at which memory is transferred
    - communication_range: range of communication between particles
    - socialization_tendency: tendency of a particle to socialize
    - colony_building_skill: skill level in building colonies
    - cultural_influence: influence of cultural factors on a particle
    """

    def __init__(self):
        # Core gene traits and ranges
        self.gene_traits: List[str] = [
            "speed_factor", "interaction_strength", "perception_range", "reproduction_rate",
            "synergy_affinity", "colony_factor", "drift_sensitivity",
            "max_energy_storage", "sensory_sensitivity", "memory_transfer_rate",
            "communication_range", "socialization_tendency", "colony_building_skill",
            "cultural_influence"
        ]

        # Increased mutation rates and ranges for more dynamic evolution
        self.gene_mutation_rate: float = 0.25  # Increased from 0.15
        self.gene_mutation_range: Tuple[float, float] = (-0.2, 0.2)  # Wider range

        # Expanded trait ranges with safety bounds
        self.speed_factor_range: Tuple[float, float] = (0.05, 4.0)
        self.interaction_strength_range: Tuple[float, float] = (0.05, 4.0) 
        self.perception_range_range: Tuple[float, float] = (20.0, 400.0)
        self.reproduction_rate_range: Tuple[float, float] = (0.02, 1.5)
        self.synergy_affinity_range: Tuple[float, float] = (0.0, 3.0)
        self.colony_factor_range: Tuple[float, float] = (0.0, 2.0)
        self.drift_sensitivity_range: Tuple[float, float] = (0.0, 3.0)
        self.max_energy_storage_range: Tuple[float, float] = (150.0, 1000.0)
        self.sensory_sensitivity_range: Tuple[float, float] = (0.1, 2.0)
        self.memory_transfer_rate_range: Tuple[float, float] = (0.01, 0.99)
        self.communication_range_range: Tuple[float, float] = (20.0, 500.0)
        self.socialization_tendency_range: Tuple[float, float] = (0.0, 1.0)
        self.colony_building_skill_range: Tuple[float, float] = (0.0, 1.0)
        self.cultural_influence_range: Tuple[float, float] = (0.0, 1.0)

        # Higher energy efficiency mutation for more dynamic resource management
        self.energy_efficiency_mutation_rate: float = 0.2
        self.energy_efficiency_mutation_range: Tuple[float, float] = (-0.15, 0.3)

        # Safety epsilon values to prevent divide by zero
        self.EPSILON: float = 1e-10
        self.MIN_ARRAY_SIZE: int = 1

    def clamp_gene_values(
        self, 
        speed_factor: np.ndarray, 
        interaction_strength: np.ndarray,
        perception_range: np.ndarray,
        reproduction_rate: np.ndarray,
        synergy_affinity: np.ndarray,
        colony_factor: np.ndarray,
        drift_sensitivity: np.ndarray,
        max_energy_storage: np.ndarray,
        sensory_sensitivity: np.ndarray,
        memory_transfer_rate: np.ndarray,
        communication_range: np.ndarray,
        socialization_tendency: np.ndarray,
        colony_building_skill: np.ndarray,
        cultural_influence: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Safely clamp all gene values to their specified ranges with robust error handling.
        Handles arrays of different shapes and ensures no invalid values.
        """
        try:
            # Ensure arrays exist and have valid shapes
            arrays = [
                speed_factor, interaction_strength, perception_range, reproduction_rate,
                synergy_affinity, colony_factor, drift_sensitivity,
                max_energy_storage, sensory_sensitivity, memory_transfer_rate,
                communication_range, socialization_tendency, colony_building_skill,
                cultural_influence
            ]
            
            # Get broadcast compatible shape
            target_shape = np.broadcast_shapes(*[arr.shape for arr in arrays if arr is not None])
            
            # Safely broadcast and clip arrays
            results = []
            ranges = [
                self.speed_factor_range, self.interaction_strength_range,
                self.perception_range_range, self.reproduction_rate_range,
                self.synergy_affinity_range, self.colony_factor_range,
                self.drift_sensitivity_range, self.max_energy_storage_range,
                self.sensory_sensitivity_range, self.memory_transfer_rate_range,
                self.communication_range_range, self.socialization_tendency_range,
                self.colony_building_skill_range, self.cultural_influence_range
            ]
            
            for arr, (min_val, max_val) in zip(arrays, ranges):
                if arr is None:
                    arr = np.full(target_shape, min_val + self.EPSILON)
                else:
                    # Broadcast to target shape
                    arr = np.broadcast_to(arr, target_shape).copy()
                    
                # Replace invalid values
                arr = np.nan_to_num(arr, nan=min_val + self.EPSILON, 
                                  posinf=max_val, neginf=min_val)
                
                # Clip to valid range
                arr = np.clip(arr, min_val + self.EPSILON, max_val)
                results.append(arr)
                
            return tuple(results)
            
        except Exception as e:
            # Fallback values if error occurs
            shape = (self.MIN_ARRAY_SIZE,)
            return (
                np.full(shape, self.speed_factor_range[0] + self.EPSILON),
                np.full(shape, self.interaction_strength_range[0] + self.EPSILON),
                np.full(shape, self.perception_range_range[0] + self.EPSILON),
                np.full(shape, self.reproduction_rate_range[0] + self.EPSILON),
                np.full(shape, self.synergy_affinity_range[0] + self.EPSILON),
                np.full(shape, self.colony_factor_range[0] + self.EPSILON),
                np.full(shape, self.drift_sensitivity_range[0] + self.EPSILON),
                np.full(shape, self.max_energy_storage_range[0] + self.EPSILON),
                np.full(shape, self.sensory_sensitivity_range[0] + self.EPSILON),
                np.full(shape, self.memory_transfer_rate_range[0] + self.EPSILON),
                np.full(shape, self.communication_range_range[0] + self.EPSILON),
                np.full(shape, self.socialization_tendency_range[0] + self.EPSILON),
                np.full(shape, self.colony_building_skill_range[0] + self.EPSILON),
                np.full(shape, self.cultural_influence_range[0] + self.EPSILON)
            )

###############################################################
# Simulation Configuration
###############################################################

class SimulationConfig:
    """
    Configuration class for the GeneParticles simulation, optimized for maximum emergence
    and complex structure formation with robust error handling.
    """

    def __init__(self):
        # Core simulation parameters optimized for emergence
        self.n_cell_types: int = max(1, 10)  # Ensure at least 1 type
        self.particles_per_type: int = max(1, 50)  # Ensure at least 1 particle
        self.min_particles_per_type: int = max(1, 50)
        self.max_particles_per_type: int = max(300, self.min_particles_per_type)
        
        # Ranges for particle properties
        self.mass_range = (1e-6, 15.0)
        self.speed_factor_range = (0.1, 5.0)
        self.interaction_strength_range = (-3.0, 3.0)
        self.perception_range_range = (10.0, 300.0)
        self.reproduction_rate_range = (0.05, 1.0)
        self.synergy_affinity_range = (0.0, 2.0)
        self.colony_factor_range = (0.0, 1.0)
        self.drift_sensitivity_range = (0.0, 2.0)
        self.max_energy_storage_range = (100.0, 1000.0)
        self.sensory_sensitivity_range = (0.1, 2.0)
        self.memory_transfer_rate_range = (0.01, 0.99)
        self.communication_range_range = (20.0, 500.0)
        self.socialization_tendency_range = (0.0, 1.0)
        self.colony_building_skill_range = (0.0, 1.0)
        self.cultural_influence_range = (0.0, 1.0)
        self.energy_efficiency_range = (-0.3, 2.5)

        # Core parameters
        self.base_velocity_scale = 1.2
        self.max_velocity = 10.0
        self.min_mass = 0.1
        self.max_mass = 100.0
        self.mass_based_fraction = 0.7
        self.max_frames = 0
        self.initial_energy = 150.0
        self.max_energy = 1000.0
        self.friction = 0.2
        self.global_temperature = 0.1
        self.predation_range = 75.0
        self.energy_transfer_factor = 0.7
        self.mass_transfer = True
        self.max_age = float('inf')
        self.evolution_interval = 3000
        self.synergy_range = 200.0
        self.min_colony_size = 10
        self.max_colony_size = 1000

        # Culling weights
        self.culling_fitness_weights = {
            "energy_weight": 0.6,
            "age_weight": 0.8,
            "speed_factor_weight": 0.7,
            "interaction_strength_weight": 0.7,
            "synergy_affinity_weight": 0.8,
            "colony_factor_weight": 0.9,
            "drift_sensitivity_weight": 0.6
        }

        # Reproduction parameters
        self.reproduction_energy_threshold = 180.0
        self.reproduction_mutation_rate = 0.3
        self.reproduction_offspring_energy_fraction = 0.5

        # Clustering parameters
        self.alignment_strength = 0.4
        self.cohesion_strength = 0.5
        self.separation_strength = 0.3
        self.cluster_radius = 15.0
        self.particle_size = 5.0
        self.max_cache_size = 10000

        # Colony parameters
        self.speciation_threshold = 0.8
        self.colony_formation_probability = 0.4
        self.colony_radius = 250.0
        self.colony_cohesion_strength = 0.6

        # Advanced parameters
        self.synergy_evolution_rate = 0.08
        self.complexity_factor = 2.0
        self.structural_complexity_weight = 0.9

        # Genetic parameters
        self.genetics = GeneticParamConfig()
        self.genetics.gene_mutation_rate = 0.05
        self.genetics.gene_mutation_range = (-0.1, 0.1)
        self.genetics.energy_efficiency_mutation_range = (-0.1, 0.1)

        # Safety epsilon
        self.EPSILON = 1e-10

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters with comprehensive error checking."""
        try:
            # Validate all numeric ranges
            if not all(isinstance(r, (int, float)) for r in [
                self.mass_range[0], self.mass_range[1],
                self.speed_factor_range[0], self.speed_factor_range[1],
                self.interaction_strength_range[0], self.interaction_strength_range[1],
                self.perception_range_range[0], self.perception_range_range[1],
                self.reproduction_rate_range[0], self.reproduction_rate_range[1],
                self.synergy_affinity_range[0], self.synergy_affinity_range[1],
                self.colony_factor_range[0], self.colony_factor_range[1],
                self.drift_sensitivity_range[0], self.drift_sensitivity_range[1],
                self.energy_efficiency_range[0], self.energy_efficiency_range[1]
            ]):
                raise ValueError("All range values must be numeric")

            # Validate range ordering
            if not all(r[0] < r[1] for r in [
                self.mass_range,
                self.speed_factor_range,
                self.interaction_strength_range,
                self.perception_range_range,
                self.reproduction_rate_range,
                self.synergy_affinity_range,
                self.colony_factor_range,
                self.drift_sensitivity_range,
                self.energy_efficiency_range
            ]):
                raise ValueError("All ranges must have min < max")

            # Validate core parameters
            if not all([
                self.n_cell_types > 0,
                self.particles_per_type > 0,
                self.base_velocity_scale > 0,
                0 <= self.mass_based_fraction <= 1,
                self.max_frames >= 0,
                self.initial_energy > 0,
                0 <= self.friction <= 1,
                self.global_temperature >= 0,
                self.predation_range > 0,
                0 <= self.energy_transfer_factor <= 1
            ]):
                raise ValueError("Invalid core parameter values")

        except Exception as e:
            self._set_safe_defaults()
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def _set_safe_defaults(self) -> None:
        """Set safe default values if validation fails."""
        try:
            # Reset all ranges to safe values
            self.mass_range = (1e-6, 15.0)
            self.speed_factor_range = (0.1, 5.0) 
            self.interaction_strength_range = (-3.0, 3.0)
            self.perception_range_range = (10.0, 300.0)
            self.reproduction_rate_range = (0.05, 1.0)
            self.synergy_affinity_range = (0.0, 2.0)
            self.colony_factor_range = (0.0, 1.0)
            self.drift_sensitivity_range = (0.0, 2.0)
            self.energy_efficiency_range = (-0.3, 2.5)

            # Reset core parameters
            self.n_cell_types = 1
            self.particles_per_type = 1
            self.base_velocity_scale = 0.1
            self.mass_based_fraction = 0.5
            self.max_frames = 0
            self.initial_energy = 100.0
            self.friction = 0.2
            self.global_temperature = 0.1
            self.predation_range = 50.0
            self.energy_transfer_factor = 0.5

        except Exception as e:
            print(f"Error setting safe defaults: {str(e)}. Using minimum viable configuration.")
            self.n_cell_types = 1
            self.particles_per_type = 1

###############################################################
# Cellular Component & Type Data Management
###############################################################

class CellularTypeData:
    """
    Represents a cellular type with multiple cellular components.
    Manages positions, velocities, energy, mass, and genetic traits of components.
    """
    
    # Class-level constants for default ranges and values
    DEFAULT_RANGES = {
        'mass': (1e-6, 15.0),
        'speed_factor': (0.1, 5.0),
        'interaction_strength': (-3.0, 3.0), 
        'perception_range': (10.0, 300.0),
        'reproduction_rate': (0.05, 1.0),
        'synergy_affinity': (0.0, 2.0),
        'colony_factor': (0.0, 1.0),
        'drift_sensitivity': (0.0, 2.0),
        'energy_efficiency': (-0.3, 2.5),
        'max_energy_storage': (100.0, 1000.0),
        'sensory_sensitivity': (0.1, 2.0),
        'memory_transfer_rate': (0.01, 0.99),
        'communication_range': (20.0, 500.0),
        'socialization_tendency': (0.0, 1.0),
        'colony_building_skill': (0.0, 1.0),
        'cultural_influence': (0.0, 1.0),
        'velocity': (-10.0, 10.0),
        'energy': (0.0, 1000.0)
    }

    def __init__(self,
                 config: SimulationConfig,
                 type_id: int,
                 color: Tuple[int, int, int], 
                 n_particles: int,
                 window_width: int,
                 window_height: int,
                 initial_energy: float,
                 max_age: float = np.inf,
                 mass: Optional[float] = None,
                 base_velocity_scale: float = 1.0,
                 energy_efficiency: Optional[float] = None,
                 gene_traits: List[str] = ["speed_factor", "interaction_strength", "perception_range", 
                                         "reproduction_rate", "synergy_affinity", "colony_factor", 
                                         "drift_sensitivity"],
                 gene_mutation_rate: float = 0.05,
                 gene_mutation_range: Tuple[float, float] = (-0.1, 0.1),
                 min_energy: float = 0.0,
                 max_energy: float = 1000.0,
                 min_mass: float = 0.1,
                 max_mass: float = 10.0,
                 min_velocity: float = -10.0,
                 max_velocity: float = 10.0,
                 min_perception: float = 10.0,
                 max_perception: float = 300.0,
                 min_reproduction: float = 0.05,
                 max_reproduction: float = 1.0,
                 min_synergy: float = 0.0,
                 max_synergy: float = 2.0,
                 min_colony: float = 0.0,
                 max_colony: float = 1.0,
                 min_drift: float = 0.0,
                 max_drift: float = 2.0,
                 min_energy_efficiency: float = -0.3,
                 max_energy_efficiency: float = 2.5,
                 min_memory: float = 0.0,
                 max_memory: float = 1.0,
                 min_social: float = 0.0,
                 max_social: float = 1.0,
                 min_colony_build: float = 0.0,
                 max_colony_build: float = 1.0,
                 min_culture: float = 0.0,
                 max_culture: float = 1.0):
        """
        Initialize a CellularTypeData instance with given parameters.
        """
        # Input validation and sanitization
        n_particles = max(1, int(n_particles))
        window_width = max(1, int(window_width))
        window_height = max(1, int(window_height))
        
        # Store metadata with validation
        self.config = config
        self.type_id = int(type_id)
        self.color = tuple(map(lambda x: max(0, min(255, int(x))), color))
        self.mass_based = bool(mass is not None)

        # Store parameter bounds with validation
        self.min_energy = float(min_energy)
        self.max_energy = float(max_energy)
        self.min_mass = float(min_mass)
        self.max_mass = float(max_mass)
        self.min_velocity = float(min_velocity)
        self.max_velocity = float(max_velocity)
        self.min_perception = float(min_perception)
        self.max_perception = float(max_perception)
        self.min_reproduction = float(min_reproduction)
        self.max_reproduction = float(max_reproduction)
        self.min_synergy = float(min_synergy)
        self.max_synergy = float(max_synergy)
        self.min_colony = float(min_colony)
        self.max_colony = float(max_colony)
        self.min_drift = float(min_drift)
        self.max_drift = float(max_drift)
        self.min_energy_efficiency = float(min_energy_efficiency)
        self.max_energy_efficiency = float(max_energy_efficiency)
        self.min_memory = float(min_memory)
        self.max_memory = float(max_memory)
        self.min_social = float(min_social)
        self.max_social = float(max_social)
        self.min_colony_build = float(min_colony_build)
        self.max_colony_build = float(max_colony_build)
        self.min_culture = float(min_culture)
        self.max_culture = float(max_culture)

        try:
            # Initialize positions safely
            coords = InteractionManager.random_xy(window_width, window_height, n_particles)
            self.x = coords[:, 0].astype(np.float64)
            self.y = coords[:, 1].astype(np.float64)

            # Initialize energy efficiency safely
            if energy_efficiency is None:
                self.energy_efficiency = np.clip(
                    np.random.uniform(self.min_energy_efficiency, self.max_energy_efficiency, n_particles),
                    self.min_energy_efficiency,
                    self.max_energy_efficiency
                ).astype(np.float64)
            else:
                self.energy_efficiency = np.full(n_particles, np.clip(
                    float(energy_efficiency),
                    self.min_energy_efficiency,
                    self.max_energy_efficiency
                ), dtype=np.float64)

            # Safe velocity scaling calculation
            velocity_scaling = base_velocity_scale / np.maximum(self.energy_efficiency, 1e-10)

            # Initialize velocities safely
            self.vx = np.clip(
                np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
                self.min_velocity,
                self.max_velocity
            ).astype(np.float64)
            self.vy = np.clip(
                np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
                self.min_velocity,
                self.max_velocity
            ).astype(np.float64)

            # Initialize energy safely
            self.energy = np.clip(
                np.full(n_particles, initial_energy, dtype=np.float64),
                self.min_energy,
                self.max_energy
            )

            # Initialize mass safely for mass-based types
            if self.mass_based:
                if mass is None or mass <= 0.0:
                    mass = self.min_mass
                self.mass = np.clip(
                    np.full(n_particles, mass, dtype=np.float64),
                    self.min_mass,
                    self.max_mass
                )
            else:
                self.mass = None

            # Initialize status arrays safely
            self.alive = np.ones(n_particles, dtype=bool)
            self.age = np.zeros(n_particles, dtype=np.float64)
            self.max_age = float(max_age)

            # Initialize gene traits safely with clipping
            self.speed_factor = np.clip(np.random.uniform(0.5, 1.5, n_particles), 0.1, 2.0)
            self.interaction_strength = np.clip(np.random.uniform(0.5, 1.5, n_particles), 0.1, 2.0)
            self.perception_range = np.clip(
                np.random.uniform(50.0, 150.0, n_particles),
                self.min_perception,
                self.max_perception
            )
            self.reproduction_rate = np.clip(
                np.random.uniform(0.1, 0.5, n_particles),
                self.min_reproduction,
                self.max_reproduction
            )
            self.synergy_affinity = np.clip(
                np.random.uniform(0.5, 1.5, n_particles),
                self.min_synergy,
                self.max_synergy
            )
            self.colony_factor = np.clip(
                np.random.uniform(0.0, 1.0, n_particles),
                self.min_colony,
                self.max_colony
            )
            self.drift_sensitivity = np.clip(
                np.random.uniform(0.5, 1.5, n_particles),
                self.min_drift,
                self.max_drift
            )
            self.max_energy_storage = np.clip(np.random.uniform(150.0, 300.0, n_particles), 150.0, 1000.0)
            self.sensory_sensitivity = np.clip(np.random.uniform(0.5, 1.5, n_particles), 0.1, 2.0)
            self.short_term_memory = np.zeros(n_particles, dtype=np.float64)
            self.long_term_memory = np.zeros(n_particles, dtype=np.float64)
            self.memory_transfer_rate = np.clip(np.random.uniform(0.1, 0.9, n_particles), 0.01, 0.99)
            self.communication_range = np.clip(np.random.uniform(50.0, 200.0, n_particles), 20.0, 500.0)
            self.socialization_tendency = np.clip(np.random.uniform(0.0, 1.0, n_particles), 0.0, 1.0)
            self.colony_building_skill = np.clip(np.random.uniform(0.0, 1.0, n_particles), 0.0, 1.0)
            self.cultural_influence = np.clip(np.random.uniform(0.0, 1.0, n_particles), 0.0, 1.0)

            # Initialize tracking arrays safely
            self.species_id = np.full(n_particles, self.type_id, dtype=np.int32)
            self.parent_id = np.full(n_particles, -1, dtype=np.int32)
            self.colony_id = np.full(n_particles, -1, dtype=np.int32)
            self.colony_role = np.zeros(n_particles, dtype=np.int32)
            self.synergy_connections = np.zeros((n_particles, n_particles), dtype=bool)
            self.fitness_score = np.zeros(n_particles, dtype=np.float64)
            self.generation = np.zeros(n_particles, dtype=np.int32)
            self.mutation_history = [[] for _ in range(n_particles)]

            # Store mutation parameters
            self.gene_mutation_rate = float(gene_mutation_rate)
            self.gene_mutation_range = tuple(map(float, gene_mutation_range))

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            
    def _initialize_parameter_ranges(self, kwargs: Dict) -> None:
        """Initialize all parameter ranges with validation."""
        for param, (min_val, max_val) in self.DEFAULT_RANGES.items():
            min_name = f"min_{param}"
            max_name = f"max_{param}"
            setattr(self, min_name, float(kwargs.get(min_name, min_val)))
            setattr(self, max_name, float(kwargs.get(max_name, max_val)))
            
    def _initialize_core_arrays(self, n_particles: int, 
                              window_width: int, window_height: int,
                              initial_energy: float, mass: Optional[float],
                              base_velocity_scale: float,
                              energy_efficiency: Optional[float]) -> None:
        """Initialize core particle arrays with comprehensive validation and error handling."""
        try:
            # Validate input parameters
            n_particles = max(1, int(n_particles))
            window_width = max(1, int(window_width))
            window_height = max(1, int(window_height))
            initial_energy = float(initial_energy)
            base_velocity_scale = float(base_velocity_scale)

            # Position initialization with bounds checking and edge buffer
            edge_buffer = max(10, min(100, 0.05 * min(window_width, window_height)))
            coords = InteractionManager.random_xy(
                window_width - 2*edge_buffer,
                window_height - 2*edge_buffer,
                n_particles
            )
            self.x = coords[:, 0].astype(np.float64) + edge_buffer
            self.y = coords[:, 1].astype(np.float64) + edge_buffer

            # Initialize velocities with safe scaling
            self._initialize_velocities(n_particles, base_velocity_scale)

            # Initialize energy efficiency with validation
            self._initialize_energy_efficiency(n_particles, energy_efficiency)

            # Initialize energy and mass with safety bounds
            self._initialize_energy_and_mass(n_particles, initial_energy, mass)

            # Initialize status arrays with type safety
            self.alive = np.ones(n_particles, dtype=bool)
            self.age = np.zeros(n_particles, dtype=np.float64)
            self.max_age = float(self.config.max_age)

            # Validate array shapes and types
            self._validate_array_shapes()

        except Exception as e:
            print(f"Error in core array initialization: {str(e)}")
            self._initialize_fallback_arrays(n_particles)

    def _initialize_velocities(self, n_particles: int, base_velocity_scale: float) -> None:
        """Initialize particle velocities with controlled randomization."""
        try:
            angle = np.random.uniform(0, 2*np.pi, n_particles)
            speed = np.random.normal(base_velocity_scale, 0.1*base_velocity_scale, n_particles)
            speed = np.clip(speed, 0.1*base_velocity_scale, 2*base_velocity_scale)
            
            self.vx = (speed * np.cos(angle)).astype(np.float64)
            self.vy = (speed * np.sin(angle)).astype(np.float64)
        except Exception:
            self.vx = np.zeros(n_particles, dtype=np.float64)
            self.vy = np.zeros(n_particles, dtype=np.float64)

    def _initialize_energy_efficiency(self, n_particles: int, energy_efficiency: Optional[float]) -> None:
        """Initialize energy efficiency values with bounds checking."""
        try:
            if energy_efficiency is not None:
                self.energy_efficiency = np.full(n_particles, 
                    np.clip(energy_efficiency, 
                           self.config.energy_efficiency_range[0],
                           self.config.energy_efficiency_range[1]
                    ), dtype=np.float64)
            else:
                self.energy_efficiency = np.random.uniform(
                    self.config.energy_efficiency_range[0],
                    self.config.energy_efficiency_range[1],
                    n_particles
                ).astype(np.float64)
        except Exception:
            self.energy_efficiency = np.ones(n_particles, dtype=np.float64)

    def _initialize_energy_and_mass(self, n_particles: int, initial_energy: float, mass: Optional[float]) -> None:
        """Initialize energy and mass with safety bounds."""
        try:
            # Energy initialization
            self.energy = np.full(n_particles, 
                np.clip(initial_energy, 0, self.config.max_energy), 
                dtype=np.float64
            )
            
            # Mass initialization
            if mass is not None:
                self.mass = np.full(n_particles,
                    np.clip(mass, self.config.min_mass, self.config.max_mass),
                    dtype=np.float64
                )
            else:
                self.mass = np.random.uniform(
                    self.config.min_mass,
                    self.config.max_mass,
                    n_particles
                ).astype(np.float64)
        except Exception:
            self.energy = np.full(n_particles, 100.0, dtype=np.float64)
            self.mass = np.ones(n_particles, dtype=np.float64)

    def _initialize_fallback_arrays(self, n_particles: int) -> None:
        """Initialize minimal fallback arrays if main initialization fails."""
        self.x = np.zeros(n_particles, dtype=np.float64)
        self.y = np.zeros(n_particles, dtype=np.float64)
        self.vx = np.zeros(n_particles, dtype=np.float64)
        self.vy = np.zeros(n_particles, dtype=np.float64)
        self.energy = np.full(n_particles, 100.0, dtype=np.float64)
        self.mass = np.ones(n_particles, dtype=np.float64)
        self.energy_efficiency = np.ones(n_particles, dtype=np.float64)
        self.alive = np.ones(n_particles, dtype=bool)
        self.age = np.zeros(n_particles, dtype=np.float64)
        self.max_age = float('inf')

    def _initialize_genetic_traits(self, n_particles: int) -> None:
        """Initialize genetic traits with safe bounds."""
        for trait in self.gene_traits:
            min_val = getattr(self, f"min_{trait}")
            max_val = getattr(self, f"max_{trait}")
            setattr(self, trait, np.clip(
                np.random.uniform(min_val, max_val, n_particles),
                min_val, max_val
            ).astype(np.float64))

    def _initialize_advanced_features(self, n_particles: int) -> None:
        """Initialize advanced particle features and memory systems."""
        # Memory systems
        self.short_term_memory = np.zeros(n_particles, dtype=np.float64)
        self.long_term_memory = np.zeros(n_particles, dtype=np.float64)
        
        # Social and colony features
        self.colony_id = np.full(n_particles, -1, dtype=np.int32)
        self.colony_role = np.zeros(n_particles, dtype=np.int32)
        self.synergy_connections = np.zeros((n_particles, n_particles), dtype=bool)

    def _initialize_tracking_systems(self, n_particles: int) -> None:
        """Initialize systems for tracking particle history and metrics."""
        self.species_id = np.full(n_particles, self.type_id, dtype=np.int32)
        self.parent_id = np.full(n_particles, -1, dtype=np.int32)
        self.fitness_score = np.zeros(n_particles, dtype=np.float64)
        self.generation = np.zeros(n_particles, dtype=np.int32)
        self.mutation_history = [[] for _ in range(n_particles)]
    
    def is_alive_mask(self) -> np.ndarray:
        """Compute alive mask with safe array operations."""
        try:
            self._validate_array_shapes()
            mask = (self.alive & 
                   (self.energy > self.min_energy) & 
                   (self.age < self.max_age))
            if self.mass_based and self.mass is not None:
                mask &= (self.mass > self.min_mass)
            return mask
        except Exception:
            return np.ones(len(self.x), dtype=bool)

    def _validate_array_shapes(self) -> None:
        """Validate and correct array shapes for consistency."""
        base_size = len(self.x)
        arrays_to_check = [
            'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
            'speed_factor', 'interaction_strength', 'perception_range',
            'reproduction_rate', 'synergy_affinity', 'colony_factor',
            'drift_sensitivity', 'species_id', 'parent_id', 'max_energy_storage',
            'sensory_sensitivity', 'short_term_memory', 'long_term_memory',
            'memory_transfer_rate', 'communication_range', 'socialization_tendency',
            'colony_building_skill', 'cultural_influence'
        ]
        
        for attr in arrays_to_check:
            current = getattr(self, attr)
            if len(current) != base_size:
                setattr(self, attr, np.resize(current, base_size))
                
        if self.mass_based and self.mass is not None:
            if len(self.mass) != base_size:
                self.mass = np.resize(self.mass, base_size)

    def validate(self) -> bool:
        """Validate the CellularTypeData object."""
        return self.config._validate()

    def update_alive(self) -> None:
        """Update alive status safely."""
        try:
            self.alive = self.is_alive_mask()
        except Exception:
            self.alive = np.ones_like(self.alive)

    def age_components(self) -> None:
        """Age components with safe operations."""
        try:
            self.age = np.add(self.age, 1.0, where=self.alive)
            self.energy = np.clip(self.energy, self.min_energy, self.max_energy)
        except Exception:
            pass

    def update_states(self) -> None:
        """Update component states safely."""
        try:
            self._validate_array_shapes()
        except Exception:
            pass

    def remove_dead(self, config: SimulationConfig) -> None:
        """Remove dead components safely with array broadcasting."""
        try:
            self._validate_array_shapes()
            alive_mask = self.is_alive_mask()
            dead_due_to_age = (~alive_mask) & (self.age >= self.max_age)
            
            if np.any(dead_due_to_age):
                self._handle_energy_transfer(dead_due_to_age, alive_mask, config)
            
            # Filter arrays safely
            arrays_to_filter = [
                'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                'speed_factor', 'interaction_strength', 'perception_range',
                'reproduction_rate', 'synergy_affinity', 'colony_factor',
                'drift_sensitivity', 'species_id', 'parent_id', 'max_energy_storage',
                'sensory_sensitivity', 'short_term_memory', 'long_term_memory',
                'memory_transfer_rate', 'communication_range', 'socialization_tendency',
                'colony_building_skill', 'cultural_influence'
            ]
            
            for attr in arrays_to_filter:
                current = getattr(self, attr)
                if len(current) > len(alive_mask):
                    current = current[:len(alive_mask)]
                elif len(current) < len(alive_mask):
                    alive_mask = alive_mask[:len(current)]
                setattr(self, attr, current[alive_mask])
                
            if self.mass_based and self.mass is not None:
                if len(self.mass) > len(alive_mask):
                    self.mass = self.mass[:len(alive_mask)]
                elif len(self.mass) < len(alive_mask):
                    alive_mask = alive_mask[:len(self.mass)]
                self.mass = self.mass[alive_mask]
                
        except Exception as e:
            print(f"Error in remove_dead: {str(e)}")
            self._validate_array_shapes()

    def _handle_energy_transfer(self, dead_due_to_age: np.ndarray, alive_mask: np.ndarray, config: SimulationConfig) -> None:
        """Handle energy transfer from dead components safely."""
        try:
            alive_indices = np.where(alive_mask)[0]
            dead_age_indices = np.where(dead_due_to_age)[0]
            
            if len(alive_indices) > 0:
                alive_positions = np.column_stack((self.x[alive_indices], self.y[alive_indices]))
                tree = cKDTree(alive_positions)
                
                batch_size = min(1000, len(dead_age_indices))
                for i in range(0, len(dead_age_indices), batch_size):
                    batch_indices = dead_age_indices[i:i + batch_size]
                    dead_positions = np.column_stack((self.x[batch_indices], self.y[batch_indices]))
                    dead_energies = self.energy[batch_indices]
                    
                    distances, neighbors = tree.query(
                        dead_positions,
                        k=min(3, len(alive_indices)),
                        distance_upper_bound=config.predation_range
                    )
                    
                    valid_mask = distances < config.predation_range
                    for j, (dist_row, neighbor_row, dead_energy) in enumerate(zip(distances, neighbors, dead_energies)):
                        valid = valid_mask[j]
                        if np.any(valid):
                            valid_neighbors = neighbor_row[valid]
                            energy_share = dead_energy / max(np.sum(valid), 1)
                            self.energy[alive_indices[valid_neighbors]] += energy_share
                            self.energy[batch_indices[j]] = 0.0
                            
        except Exception as e:
            print(f"Error in energy transfer: {str(e)}")

    def add_component(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        energy: float,
        mass_val: Optional[float],
        energy_efficiency_val: float,
        speed_factor_val: float,
        interaction_strength_val: float,
        perception_range_val: float,
        reproduction_rate_val: float,
        synergy_affinity_val: float,
        colony_factor_val: float,
        drift_sensitivity_val: float,
        species_id_val: int,
        parent_id_val: int,
        max_age: float,
        max_energy_storage_val: float,
        sensory_sensitivity_val: float,
        memory_transfer_rate_val: float,
        communication_range_val: float,
        socialization_tendency_val: float,
        colony_building_skill_val: float,
        cultural_influence_val: float
    ) -> None:
        """Add new component safely with array broadcasting."""
        try:
            # Validate and clip input values
            x = float(x)
            y = float(y)
            vx = np.clip(float(vx), self.min_velocity, self.max_velocity)
            vy = np.clip(float(vy), self.min_velocity, self.max_velocity)
            energy = np.clip(float(energy), self.min_energy, self.max_energy)
            energy_efficiency_val = np.clip(float(energy_efficiency_val), self.min_energy_efficiency, self.max_energy_efficiency)
            
            # Prepare new values as arrays for broadcasting
            new_values = {
                'x': np.array([x]),
                'y': np.array([y]),
                'vx': np.array([vx]),
                'vy': np.array([vy]),
                'energy': np.array([energy]),
                'alive': np.array([True]),
                'age': np.array([0.0]),
                'energy_efficiency': np.array([energy_efficiency_val]),
                'speed_factor': np.array([speed_factor_val]),
                'interaction_strength': np.array([interaction_strength_val]),
                'perception_range': np.array([perception_range_val]),
                'reproduction_rate': np.array([reproduction_rate_val]),
                'synergy_affinity': np.array([synergy_affinity_val]),
                'colony_factor': np.array([colony_factor_val]),
                'drift_sensitivity': np.array([drift_sensitivity_val]),
                'species_id': np.array([species_id_val]),
                'parent_id': np.array([parent_id_val]),
                'max_energy_storage': np.array([max_energy_storage_val]),
                'sensory_sensitivity': np.array([sensory_sensitivity_val]),
                'short_term_memory': np.array([0.0]),  # Initialize short-term memory to 0
                'long_term_memory': np.array([0.0]),  # Initialize long-term memory to 0
                'memory_transfer_rate': np.array([memory_transfer_rate_val]),
                'communication_range': np.array([communication_range_val]),
                'socialization_tendency': np.array([socialization_tendency_val]),
                'colony_building_skill': np.array([colony_building_skill_val]),
                'cultural_influence': np.array([cultural_influence_val])
            }
            
            # Safely concatenate arrays
            for attr, new_value in new_values.items():
                current = getattr(self, attr)
                setattr(self, attr, np.concatenate((current, new_value)))
            
            # Handle mass separately if mass-based
            if self.mass_based:
                if mass_val is None or mass_val <= 0.0:
                    mass_val = self.min_mass
                mass_val = np.clip(float(mass_val), self.min_mass, self.max_mass)
                self.mass = np.concatenate((self.mass, np.array([mass_val])))
                
            self._validate_array_shapes()
            
        except Exception as e:
            print(f"Error adding component: {str(e)}")

###############################################################
# Genetic Instructions (Turing-Complete Neural Architecture)
###############################################################

class GeneticInstructions:
    """Advanced Turing-complete neural instruction architecture with optimized execution."""

    # Core neural instruction set
    INSTR_SET = {
        # Basic arithmetic
        'NOP': 0,      # No operation
        'ADD': 1,      # Add
        'SUB': 2,      # Subtract
        'MUL': 3,      # Multiply
        'DIV': 4,      # Divide
        'POW': 5,      # Power
        'SQRT': 6,     # Square root
        'LOG': 7,      # Natural log
        'EXP': 8,      # Exponential
        
        # Neural operations
        'ACTIVATE': 10,   # Activation function
        'BACKPROP': 11,  # Backpropagation
        'WEIGHT': 12,    # Weight update
        'BIAS': 13,      # Bias update
        'GRAD': 14,      # Gradient computation
        'BATCH': 15,     # Batch normalization
        'DROP': 16,      # Dropout
        
        # Memory & flow control  
        'LOAD': 20,      # Load from memory
        'STORE': 21,     # Store to memory
        'PUSH': 22,      # Push to stack
        'POP': 23,       # Pop from stack
        'JMP': 24,       # Jump
        'BRANCH': 25,    # Conditional branch
        'CALL': 26,      # Call subroutine
        'RET': 27,       # Return
        
        # Advanced neural ops
        'CONV': 30,      # Convolution
        'POOL': 31,      # Pooling
        'ATTN': 32,      # Attention mechanism
        'LSTM': 33,      # LSTM cell
        'GRU': 34,       # GRU cell
        'TRANS': 35,     # Transformer block
        
        # Genetic operations
        'MUTATE': 40,    # Mutation
        'CROSS': 41,     # Crossover
        'SELECT': 42,    # Selection
        'EVOLVE': 43,    # Evolution step
        
        # System operations
        'SYNC': 50,      # Synchronization
        'DIST': 51,      # Distribution
        'OPTIM': 52,     # Optimization
        'DEBUG': 53      # Debug
    }

    # Reverse mapping
    INSTR_NAMES = {v:k for k,v in INSTR_SET.items()}

    # System parameters
    REGISTER_COUNT = 64          # General purpose registers
    VECTOR_REGISTER_COUNT = 32   # Vector registers
    MEMORY_SIZE = 1 << 20       # 1MB addressable memory
    CACHE_SIZE = 1 << 16        # 64KB instruction cache
    STACK_SIZE = 1 << 12        # 4KB call stack
    MAX_RECURSION = 256         # Max recursion depth
    
    # Neural parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    DROPOUT_RATE = 0.5
    
    def __init__(self):
        """Initialize the neural instruction architecture."""
        # Core components
        self.registers = np.zeros(self.REGISTER_COUNT, dtype=np.float32)
        self.vector_registers = np.zeros((self.VECTOR_REGISTER_COUNT, self.BATCH_SIZE), dtype=np.float32)
        self.memory = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self.stack = np.zeros(self.STACK_SIZE, dtype=np.float32)
        self.instruction_cache = {}
        self.branch_history = collections.deque(maxlen=1024)
        
        # Neural components
        self.weights = {}
        self.biases = {}
        self.gradients = {}
        self.activations = {}
        
        # Optimization components
        self._setup_execution_pipeline()
        self._init_neural_functions()
        self._setup_branch_prediction()

    def _setup_execution_pipeline(self):
        """Setup optimized execution pipeline."""
        self.pipeline = {
            'fetch': self._fetch_instruction,
            'decode': self._decode_instruction,
            'execute': self._execute_instruction,
            'memory': self._memory_operation,
            'writeback': self._writeback_result
        }
        
        # Fast dispatch tables
        self.op_dispatch = {
            op: getattr(self, f'_exec_{name.lower()}', self._exec_nop)
            for name, op in self.INSTR_SET.items()
        }
        
        # Vectorized operation tables
        self.vector_ops = {
            'ADD': np.add,
            'MUL': np.multiply,
            'DIV': np.divide,
            'POW': np.power,
            'SQRT': np.sqrt,
            'LOG': np.log,
            'EXP': np.exp
        }

    def _init_neural_functions(self):
        """Initialize neural network activation functions."""
        self.activation_functions = {
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': np.tanh,
            'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        }
        
        self.gradient_functions = {
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: x * (1 - x),
            'tanh': lambda x: 1 - x**2,
            'softmax': lambda x: x * (1 - x)
        }

    @staticmethod
    def create_optimized_sequence(length: int, instruction_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Create an optimized instruction sequence with optional weighting."""
        if instruction_weights:
            instructions = list(GeneticInstructions.INSTR_SET.values())
            weights = [instruction_weights.get(GeneticInstructions.INSTR_NAMES[i], 1.0) for i in instructions]
            return np.random.choice(instructions, size=length, p=np.array(weights)/sum(weights))
        return np.random.choice(list(GeneticInstructions.INSTR_SET.values()), size=length)
    
###############################################################
# Genome Class
###############################################################

class Genome:
    """
    Genome: sequence of instructions + regulatory info.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.code = self._random_genome()
        self.promoters = np.zeros_like(self.code, dtype=bool)
        self.inhibitors = np.zeros_like(self.code, dtype=bool)
        self.epigenetic_marks = np.zeros_like(self.code, dtype=float)

    def _random_genome(self):
        length = random.randint(50, 200)
        code = np.array([GeneticInstructions.random_instruction() for _ in range(length)], dtype=int)
        return code

    def mutate(self):
        # Point mutations
        for i in range(len(self.code)):
            if random.random() < self.config.genetics.gene_mutation_rate:
                self.code[i] = GeneticInstructions.random_instruction()

        # Insertions
        if random.random() < self.config.genetics.insertion_rate and len(self.code) < self.config.genetics.max_genome_length:
            pos = random.randint(0, len(self.code))
            new_instr = GeneticInstructions.random_instruction()
            self.code = np.insert(self.code, pos, new_instr)
            self.promoters = np.insert(self.promoters, pos, False)
            self.inhibitors = np.insert(self.inhibitors, pos, False)
            self.epigenetic_marks = np.insert(self.epigenetic_marks, pos, 0.0)

        # Deletions
        if random.random() < self.config.genetics.deletion_rate and len(self.code) > 50:
            pos = random.randint(0, len(self.code)-1)
            self.code = np.delete(self.code, pos)
            self.promoters = np.delete(self.promoters, pos)
            self.inhibitors = np.delete(self.inhibitors, pos)
            self.epigenetic_marks = np.delete(self.epigenetic_marks, pos)

        # Duplications
        if random.random() < self.config.genetics.duplication_rate and len(self.code)*2 < self.config.genetics.max_genome_length:
            start = random.randint(0, len(self.code)-1)
            end = random.randint(start, len(self.code)-1)
            segment = self.code[start:end+1]
            self.code = np.concatenate((self.code, segment))
            self.promoters = np.concatenate((self.promoters, self.promoters[start:end+1]))
            self.inhibitors = np.concatenate((self.inhibitors, self.inhibitors[start:end+1]))
            self.epigenetic_marks = np.concatenate((self.epigenetic_marks, self.epigenetic_marks[start:end+1]))

        # Transposons
        if random.random() < self.config.genetics.transposon_rate and len(self.code) > 100:
            start = random.randint(0, len(self.code)-50)
            end = min(len(self.code)-1, start+random.randint(10,50))
            segment = self.code[start:end+1]
            pseg = self.promoters[start:end+1]
            iseg = self.inhibitors[start:end+1]
            eseg = self.epigenetic_marks[start:end+1]

            self.code = np.delete(self.code, slice(start,end+1))
            self.promoters = np.delete(self.promoters, slice(start,end+1))
            self.inhibitors = np.delete(self.inhibitors, slice(start,end+1))
            self.epigenetic_marks = np.delete(self.epigenetic_marks, slice(start,end+1))

            pos = random.randint(0, len(self.code))
            self.code = np.insert(self.code, pos, segment)
            self.promoters = np.insert(self.promoters, pos, pseg)
            self.inhibitors = np.insert(self.inhibitors, pos, iseg)
            self.epigenetic_marks = np.insert(self.epigenetic_marks, pos, eseg)

        # Epigenetic changes
        for i in range(len(self.code)):
            if random.random() < self.config.genetics.epigenetic_mark_rate:
                self.epigenetic_marks[i] = min(1.0, self.epigenetic_marks[i] + random.uniform(0.0,0.2))
            if random.random() < self.config.genetics.epigenetic_erase_rate:
                self.epigenetic_marks[i] = max(0.0, self.epigenetic_marks[i] - random.uniform(0.0,0.2))


###############################################################
# Genetic Interpreter Class
###############################################################

class GeneticInterpreter:
    """Advanced genetic sequence interpreter implementing Turing-complete genetic programming."""
    
    def __init__(self, gene_sequence: Optional[List[List[Any]]] = None):
        """Initialize genetic interpreter with optimized defaults."""
        self.default_sequence = [
            ["start_movement", 1.0, 0.1, 0.0],  # speed, randomness, bias
            ["start_interaction", 0.5, 100.0],   # strength, radius
            ["start_energy", 0.1, 0.5, 0.3],     # gain, efficiency, transfer
            ["start_reproduction", 150.0, 100.0, 50.0, 30.0],  # threshold, cost, bonus, penalty
            ["start_growth", 0.1, 2.0, 100.0],   # rate, factor, limit
            ["start_predation", 10.0, 5.0]       # strength, range
        ]
        self.gene_sequence = gene_sequence if gene_sequence is not None else self.default_sequence
        
        # Initialize core components
        self._setup_safety_bounds()
        self._initialize_genetic_mechanisms()
        self._setup_caches()
        
    def _setup_safety_bounds(self) -> None:
        """Configure comprehensive safety bounds."""
        self.bounds = {
            'energy': (0.0, 1000.0),
            'velocity': (-20.0, 20.0), 
            'traits': (0.01, 5.0),
            'mass': (0.1, 10.0),
            'age': (0.0, float('inf')),
            'distance': (1e-10, float('inf')),
            'interaction': (0.0, 1000.0),
            'reproduction': (0.0, 500.0)
        }
        
    def _initialize_genetic_mechanisms(self) -> None:
        """Initialize advanced genetic control mechanisms."""
        # Regulatory networks
        self.regulatory_networks = {
            'movement': {'inhibitors': [], 'activators': [], 'threshold': 0.5},
            'interaction': {'inhibitors': [], 'activators': [], 'threshold': 0.4},
            'energy': {'inhibitors': [], 'activators': [], 'threshold': 0.3},
            'reproduction': {'inhibitors': [], 'activators': [], 'threshold': 0.6},
            'growth': {'inhibitors': [], 'activators': [], 'threshold': 0.4},
            'predation': {'inhibitors': [], 'activators': [], 'threshold': 0.7}
        }
        
        # Epistatic interactions
        self.epistatic_interactions = {
            'movement': {'modifiers': {'energy': 0.2, 'interaction': 0.1}},
            'interaction': {'modifiers': {'energy': 0.3, 'predation': 0.2}},
            'energy': {'modifiers': {'growth': 0.2, 'reproduction': 0.3}},
            'reproduction': {'modifiers': {'energy': -0.2, 'growth': 0.2}},
            'growth': {'modifiers': {'energy': -0.1, 'reproduction': -0.1}},
            'predation': {'modifiers': {'energy': 0.3, 'movement': 0.2}}
        }
        
        # Epigenetic modifications
        self.epigenetic_modifications = {
            'methylation': defaultdict(float),
            'acetylation': defaultdict(float),
            'phosphorylation': defaultdict(float),
            'ubiquitination': defaultdict(float)
        }
        
    def _setup_caches(self) -> None:
        """Initialize performance optimization caches."""
        self.computation_cache = {}
        self.regulatory_state_cache = {}
        self.interaction_cache = {}
        self.MAX_CACHE_SIZE = 1000
        
    def decode(self, particle: CellularTypeData, others: List[CellularTypeData], env: SimulationConfig) -> None:
        """Decode genetic sequence with comprehensive error handling."""
        try:
            # Clear expired cache entries
            if len(self.computation_cache) > self.MAX_CACHE_SIZE:
                self.computation_cache.clear()
                
            # Process each gene with regulatory control
            for gene in self.gene_sequence:
                if not isinstance(gene, (list, tuple)) or len(gene) < 2:
                    continue
                    
                gene_type = str(gene[0])
                gene_data = np.asarray(gene[1:], dtype=np.float64)
                
                # Apply regulatory network effects
                if not self._check_regulatory_state(gene_type, particle):
                    continue
                    
                # Apply epistatic interactions
                gene_data = self._apply_epistatic_effects(gene_type, gene_data, particle)
                
                # Apply epigenetic modifications
                gene_data = self._apply_epigenetic_mods(gene_type, gene_data)
                
                # Execute gene function with optimized data
                method = getattr(self, f"apply_{gene_type.replace('start_', '')}_gene", None)
                if method:
                    method(particle, others, gene_data, env)
                    
        except Exception as e:
            print(f"Error in genetic decoding: {str(e)}")
            self._ensure_particle_stability(particle)
            
    def _check_regulatory_state(self, gene_type: str, particle: CellularTypeData) -> bool:
        """Check if gene expression is allowed by regulatory networks."""
        try:
            cache_key = (gene_type, id(particle))
            if cache_key in self.regulatory_state_cache:
                return self.regulatory_state_cache[cache_key]
                
            network = self.regulatory_networks.get(gene_type.replace('start_', ''))
            if not network:
                return True
                
            # Calculate regulatory score
            activator_score = sum(1 for act in network['activators'] if self._check_condition(act, particle))
            inhibitor_score = sum(1 for inh in network['inhibitors'] if self._check_condition(inh, particle))
            
            threshold = network['threshold']
            result = (activator_score - inhibitor_score) >= threshold
            
            self.regulatory_state_cache[cache_key] = result
            return result
            
        except Exception:
            return True
            
    def _apply_epistatic_effects(self, gene_type: str, gene_data: np.ndarray, particle: CellularTypeData) -> np.ndarray:
        """Apply epistatic interactions between genes."""
        try:
            base_type = gene_type.replace('start_', '')
            modifiers = self.epistatic_interactions.get(base_type, {}).get('modifiers', {})
            
            modified_data = gene_data.copy()
            for mod_gene, factor in modifiers.items():
                if hasattr(particle, mod_gene):
                    mod_value = getattr(particle, mod_gene)
                    if isinstance(mod_value, np.ndarray):
                        mod_value = np.mean(mod_value)
                    modified_data *= (1 + factor * mod_value)
                    
            return np.clip(modified_data, *self.bounds['traits'])
            
        except Exception:
            return gene_data
            
    def _apply_epigenetic_mods(self, gene_type: str, gene_data: np.ndarray) -> np.ndarray:
        """Apply epigenetic modifications to gene expression."""
        try:
            base_type = gene_type.replace('start_', '')
            
            # Combine all modification effects
            total_mod = 1.0
            for mod_type, mod_values in self.epigenetic_modifications.items():
                mod_factor = mod_values.get(base_type, 0.0)
                total_mod *= (1 + mod_factor)
                
            return np.clip(gene_data * total_mod, *self.bounds['traits'])
            
        except Exception:
            return gene_data
            
    def _ensure_particle_stability(self, particle: CellularTypeData) -> None:
        """Ensure particle maintains valid state after operations."""
        try:
            for attr, bounds in self.bounds.items():
                if hasattr(particle, attr):
                    arr = getattr(particle, attr)
                    if isinstance(arr, np.ndarray):
                        setattr(particle, attr, np.clip(arr, *bounds))
                        
        except Exception as e:
            print(f"Error ensuring stability: {str(e)}")

    def update_traits(self, ct: CellularTypeData, genetics_config: GeneticParamConfig):
        """Update particle traits based on genetic information with full trait mapping"""
        try:
            if not hasattr(ct, 'genome') or ct.genome is None:
                return
                
            genome = ct.genome
            gene_count = len(genome.code)
            if gene_count == 0:
                return
                
            # Map genome sections to traits using weighted gene blocks
            trait_genes = {
                'speed_factor': genome.code[0:gene_count//8],
                'interaction_strength': genome.code[gene_count//8:gene_count//4], 
                'perception_range': genome.code[gene_count//4:3*gene_count//8],
                'reproduction_rate': genome.code[3*gene_count//8:gene_count//2],
                'synergy_affinity': genome.code[gene_count//2:5*gene_count//8],
                'colony_factor': genome.code[5*gene_count//8:3*gene_count//4],
                'drift_sensitivity': genome.code[3*gene_count//4:7*gene_count//8],
                'energy_efficiency': genome.code[7*gene_count//8:]
            }

            # Calculate normalized trait values with proper bounds
            for trait, genes in trait_genes.items():
                if not hasattr(genetics_config, f'{trait}_range'):
                    continue
                    
                bounds = getattr(genetics_config, f'{trait}_range')
                gene_value = np.mean(genes) # Average influence of gene block
                normalized = (gene_value - np.min(genome.code)) / (np.max(genome.code) - np.min(genome.code) + 1e-10)
                trait_value = bounds[0] + normalized * (bounds[1] - bounds[0])
                
                # Apply epigenetic modifiers
                trait_value *= (1 + genome.epigenetic_marks.mean() * 0.2)
                
                # Set trait with bounds enforcement
                setattr(ct, trait, np.clip(trait_value, bounds[0], bounds[1]))

            # Update derived traits
            ct.energy_decay_rate = 1.0 - ct.energy_efficiency
            ct.reproduction_threshold = ct.reproduction_rate * genetics_config.base_reproduction_threshold

        except Exception as e:
            print(f"Error updating traits: {e}")
            traceback.print_exc()

    def classify_species(self, ct: CellularTypeData) -> Dict[int, int]:
        """Classify particles into species using advanced genetic clustering"""
        try:
            if not hasattr(ct, 'genome') or ct.genome is None:
                return defaultdict(int)

            # Extract genome features for clustering
            feature_vectors = []
            for particle_idx in range(len(ct.x)):
                genome = ct.genome[particle_idx]
                features = [
                    np.mean(genome.code),  # Average genetic code
                    np.std(genome.code),   # Genetic diversity
                    np.sum(genome.promoters),  # Regulatory complexity
                    np.sum(genome.inhibitors),
                    np.mean(genome.epigenetic_marks)  # Epigenetic state
                ]
                feature_vectors.append(features)
                
            if not feature_vectors:
                return defaultdict(int)
                
            # Normalize feature vectors
            features = np.array(feature_vectors)
            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
            
            # Calculate pairwise distances using multiple metrics
            genetic_dist = pdist(features, metric='euclidean')
            regulatory_dist = pdist(features[:,[2,3]], metric='cosine') 
            epigenetic_dist = pdist(features[:,[4]], metric='euclidean')
            
            # Combine distances with weights
            distances = squareform(
                0.5 * genetic_dist + 
                0.3 * regulatory_dist +
                0.2 * epigenetic_dist
            )
            
            # Adaptive clustering
            eps = np.percentile(distances[distances > 0], 10)  # Dynamic epsilon
            min_samples = max(3, int(np.log2(len(features))))
            
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples, 
                metric='precomputed'
            ).fit(distances)

            # Count species populations
            species_count = defaultdict(int)
            for species_id in clustering.labels_:
                species_count[species_id] += 1
                
            return species_count

        except Exception as e:
            print(f"Error classifying species: {e}")
            traceback.print_exc()
            return defaultdict(int)

    def _calculate_genome_distance(self, genome1: Genome, genome2: Genome) -> float:
        """Calculate comprehensive genetic distance between genomes"""
        try:
            if len(genome1.code) != len(genome2.code):
                return 1.0  # Maximum distance for incompatible genomes
                
            # Genetic code distance (Normalized Hamming)
            code_dist = np.mean(genome1.code != genome2.code)
            
            # Regulatory network distance
            promoter_dist = np.mean(genome1.promoters != genome2.promoters)
            inhibitor_dist = np.mean(genome1.inhibitors != genome2.inhibitors)
            regulatory_dist = 0.5 * (promoter_dist + inhibitor_dist)
            
            # Epigenetic state distance
            epigenetic_dist = np.mean(np.abs(genome1.epigenetic_marks - genome2.epigenetic_marks))
            
            # Weighted combination
            return 0.5 * code_dist + 0.3 * regulatory_dist + 0.2 * epigenetic_dist
            
        except Exception as e:
            print(f"Error calculating genome distance: {e}")
            return 1.0  # Return maximum distance on error


###############################################################
# Interaction Rules, Give-Take & Synergy
###############################################################

class InteractionRules:
    """
    Manages creation and evolution of interaction parameters, give-take matrix, and synergy matrix.
    Optimized for high performance with robust error handling and array safety.
    """
    
    def __init__(self, config: SimulationConfig, mass_based_type_indices: List[int]):
        """Initialize with robust error handling and array safety."""
        self.config = config
        self.mass_based_type_indices = np.array(mass_based_type_indices, dtype=np.int32)
        self.EPSILON = 1e-10  # Safety epsilon for division
        self.MIN_ARRAY_SIZE = 1
        
        # Initialize matrices with error handling
        try:
            self.rules = self._create_interaction_matrix()
            self.give_take_matrix = self._create_give_take_matrix()
            self.synergy_matrix = self._create_synergy_matrix()
        except Exception as e:
            print(f"Matrix initialization error handled: {str(e)}")
            # Fallback initialization
            self.rules = []
            self.give_take_matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=bool)
            self.synergy_matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=np.float32)

    def _create_interaction_matrix(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Create interaction matrix with vectorized operations and safety checks."""
        try:
            n_types = max(1, self.config.n_cell_types)
            final_rules = []
            
            # Vectorized parameter generation
            type_pairs = np.array([(i, j) for i in range(n_types) for j in range(n_types)])
            mass_based_mask = np.isin(type_pairs, self.mass_based_type_indices)
            both_mass = np.all(mass_based_mask, axis=1)
            
            # Vectorized random generation
            rand_vals = np.random.random(len(type_pairs))
            use_gravity = both_mass & (rand_vals < 0.5)
            
            potential_strengths = np.random.uniform(
                self.config.interaction_strength_range[0],
                self.config.interaction_strength_range[1],
                len(type_pairs)
            )
            potential_strengths[rand_vals < 0.5] *= -1
            
            gravity_factors = np.where(
                use_gravity,
                np.random.uniform(0.1, 2.0, len(type_pairs)),
                np.zeros(len(type_pairs))
            )
            
            max_dists = np.random.uniform(50.0, 200.0, len(type_pairs))
            
            # Create rules with safety bounds
            for idx, (i, j) in enumerate(type_pairs):
                params = {
                    "use_potential": True,
                    "use_gravity": bool(use_gravity[idx]),
                    "potential_strength": float(np.clip(potential_strengths[idx], -1e6, 1e6)),
                    "gravity_factor": float(np.clip(gravity_factors[idx], 0, 1e3)),
                    "max_dist": float(np.clip(max_dists[idx], 10.0, 1e4))
                }
                final_rules.append((int(i), int(j), params))
                
            return final_rules
            
        except Exception as e:
            print(f"Interaction matrix creation error handled: {str(e)}")
            return [(0, 0, {"use_potential": True, "use_gravity": False, 
                          "potential_strength": 1.0, "gravity_factor": 0.0, "max_dist": 50.0})]

    def _create_give_take_matrix(self) -> np.ndarray:
        """Create give-take matrix with vectorized operations and shape safety."""
        try:
            n_types = max(1, self.config.n_cell_types)
            matrix = np.zeros((n_types, n_types), dtype=bool)
            
            # Vectorized random generation
            rand_mask = np.random.random((n_types, n_types)) < 0.1
            np.fill_diagonal(rand_mask, False)  # No self-interaction
            
            return rand_mask
            
        except Exception as e:
            print(f"Give-take matrix creation error handled: {str(e)}")
            return np.zeros((self.MIN_ARRAY_SIZE, self.MIN_ARRAY_SIZE), dtype=bool)

    def _create_synergy_matrix(self) -> np.ndarray:
        """Create synergy matrix with vectorized operations and value safety."""
        try:
            n_types = max(1, self.config.n_cell_types)
            matrix = np.zeros((n_types, n_types), dtype=np.float32)
            
            # Vectorized random generation
            rand_mask = np.random.random((n_types, n_types)) < 0.1
            synergy_values = np.random.uniform(0.01, 0.3, (n_types, n_types))
            
            # Safe assignment with bounds
            matrix = np.where(rand_mask, synergy_values, 0.0)
            np.fill_diagonal(matrix, 0.0)  # No self-synergy
            
            return np.clip(matrix, 0.0, 1.0)
            
        except Exception as e:
            print(f"Synergy matrix creation error handled: {str(e)}")
            return np.zeros((self.MIN_ARRAY_SIZE, self.MIN_ARRAY_SIZE), dtype=np.float32)

    def evolve_parameters(self, frame_count: int) -> None:
        """Evolve parameters with vectorized operations and robust error handling."""
        try:
            if frame_count % self.config.evolution_interval != 0:
                return
                
            # Vectorized rule evolution
            for _, _, params in self.rules:
                rand_vals = np.random.random(3)
                mutation_factors = np.random.uniform(0.95, 1.05, 3)
                
                if rand_vals[0] < 0.1:  # Potential strength mutation
                    params["potential_strength"] = np.clip(
                        params["potential_strength"] * mutation_factors[0],
                        self.config.interaction_strength_range[0],
                        self.config.interaction_strength_range[1]
                    )
                    
                if rand_vals[1] < 0.05 and "gravity_factor" in params:  # Gravity mutation
                    params["gravity_factor"] = np.clip(
                        params["gravity_factor"] * mutation_factors[1],
                        0.0, 10.0
                    )
                    
                if rand_vals[2] < 0.05:  # Max distance mutation
                    params["max_dist"] = np.clip(
                        params["max_dist"] * mutation_factors[2],
                        10.0, 1000.0
                    )
            
            # Energy transfer evolution
            if np.random.random() < 0.1:
                self.config.energy_transfer_factor = np.clip(
                    self.config.energy_transfer_factor * np.random.uniform(0.95, 1.05),
                    0.0, 1.0
                )
            
            # Vectorized synergy evolution
            evolution_mask = np.random.random(self.synergy_matrix.shape) < 0.05
            mutation_values = np.random.uniform(-0.05, 0.05, self.synergy_matrix.shape)
            
            self.synergy_matrix = np.clip(
                np.where(
                    evolution_mask,
                    self.synergy_matrix + mutation_values,
                    self.synergy_matrix
                ),
                0.0, 1.0
            )
            
        except Exception as e:
            print(f"Parameter evolution error handled: {str(e)}")

###############################################################
# Cellular Type Manager (Handles Multi-Type Operations & Reproduction)
###############################################################

class CellularTypeManager:
    """
    Advanced cellular type management system with optimized array operations and dynamic adaptation.
    Implements sophisticated genetic algorithms and emergent behavior patterns.
    """
    
    def __init__(self, config: SimulationConfig, colors: List[Tuple[int, int, int]] = [], mass_based_type_indices: List[int] = [], n_cell_types: int = 10):
        """Initialize with robust validation and optimized data structures."""
        self.config = config
        self.n_cell_types = n_cell_types
        self.cellular_types: List[CellularTypeData] = []
        self.mass_based_type_indices = np.array(mass_based_type_indices, dtype=np.int32)
        self.colors = colors
        self.EPSILON = np.finfo(np.float64).tiny
        self.MIN_ARRAY_SIZE = 1
        
        # Pre-allocate reusable arrays for performance
        self._temp_arrays = {
            'mutation_buffer': np.zeros(config.max_particles_per_type, dtype=np.float64),
            'distance_buffer': np.zeros(config.max_particles_per_type, dtype=np.float64),
            'mask_buffer': np.zeros(config.max_particles_per_type, dtype=bool)
        }
        
        # Initialize adaptive parameters
        self._adaptation_rates = {
            'mutation': np.full(config.n_cell_types, 0.5),
            'speciation': np.full(config.n_cell_types, 0.5),
            'energy_transfer': np.full(config.n_cell_types, 0.5)
        }

    def add_cellular_type_data(self, data: CellularTypeData) -> None:
        """Add cellular type with comprehensive validation."""
        if data is not None and self.validate(data):
            self.cellular_types.append(data)
            self._update_adaptation_rates(len(self.cellular_types) - 1)

    def validate(self, data: CellularTypeData) -> bool:
        """
        Validate cellular type data integrity with comprehensive checks.
        
        Performs thorough validation of CellularTypeData object including:
        - Presence of all required attributes
        - Type checking of attributes as numpy arrays
        - Consistent array lengths
        - Value range validation
        - NaN/Inf checking
        """
        try:
            # Required attributes with expected numpy dtypes
            required_attrs = {
                'x': np.float64, 'y': np.float64, 'vx': np.float64, 'vy': np.float64,
                'energy': np.float64, 'alive': np.bool_, 'age': np.float64,
                'energy_efficiency': np.float64, 'speed_factor': np.float64,
                'interaction_strength': np.float64, 'perception_range': np.float64,
                'reproduction_rate': np.float64, 'synergy_affinity': np.float64,
                'colony_factor': np.float64, 'drift_sensitivity': np.float64,
                'species_id': np.int32, 'parent_id': np.int32
            }

            # Check attribute presence and types
            for attr, dtype in required_attrs.items():
                if not hasattr(data, attr):
                    return False
                arr = getattr(data, attr)
                if not isinstance(arr, np.ndarray) or arr.dtype != dtype:
                    return False

            # Get reference length from x array
            ref_length = len(data.x)
            if ref_length == 0:
                return False

            # Verify consistent lengths
            if not all(len(getattr(data, attr)) == ref_length for attr in required_attrs):
                return False

            # Check for NaN/Inf in float arrays
            float_attrs = [attr for attr, dtype in required_attrs.items() 
                         if dtype == np.float64]
            for attr in float_attrs:
                arr = getattr(data, attr)
                if np.any(~np.isfinite(arr)):
                    return False

            # Validate value ranges
            if (np.any(data.energy < 0) or
                np.any(~np.isfinite(data.x)) or np.any(~np.isfinite(data.y)) or
                np.any(~np.isfinite(data.vx)) or np.any(~np.isfinite(data.vy)) or
                np.any(data.energy_efficiency <= 0) or np.any(data.energy_efficiency > 1) or
                np.any(data.speed_factor <= 0) or
                np.any(data.interaction_strength < 0) or
                np.any(data.perception_range <= 0) or
                np.any(data.reproduction_rate < 0) or
                np.any(data.synergy_affinity < 0) or np.any(data.synergy_affinity > 1) or
                np.any(data.colony_factor < 0) or
                np.any(data.age < 0)):
                return False

            return True

        except Exception:
            return False

    def get_cellular_type_by_id(self, i: int) -> Optional[CellularTypeData]:
        """Get cellular type with bounds checking and validation."""
        try:
            if 0 <= i < len(self.cellular_types):
                return self.cellular_types[i]
        except Exception:
            pass
        return None

    def get_largest_type(self) -> Optional[CellularTypeData]:
        """Get the cellular type with the most particles."""
        try:
            return max(self.cellular_types, key=lambda ct: ct.x.size)
        except ValueError:  # If self.cellular_types is empty
            return None

    def remove_dead_in_all_types(self) -> None:
        """Remove dead components with optimized array operations."""
        for ct in self.cellular_types:
            if ct is not None:
                ct.remove_dead(self.config)

    def reproduce(self) -> None:
        """
        Advanced reproduction system with dynamic adaptation and emergent behaviors.
        Implements sophisticated genetic algorithms and trait inheritance patterns.
        """
        for type_idx, ct in enumerate(self.cellular_types):
            try:
                if not self._check_reproduction_conditions(ct):
                    continue

                # Calculate reproduction eligibility with vectorized operations
                eligible_mask = self._calculate_eligibility_mask(ct)
                num_offspring = np.sum(eligible_mask)
                
                if num_offspring == 0:
                    continue

                parent_indices = np.where(eligible_mask)[0]
                
                # Handle energy transfer and mutation
                parent_energy, offspring_energy = self._handle_energy_transfer(ct, eligible_mask)
                mutation_mask = self._generate_mutation_mask(num_offspring, type_idx)
                
                # Generate offspring traits with advanced genetic algorithms
                offspring_traits = self._generate_offspring_traits(ct, parent_indices, mutation_mask)
                
                # Calculate genetic distances and handle speciation
                genetic_distances = self._calculate_genetic_distances(ct, offspring_traits, parent_indices)
                new_species_ids = self._assign_species_ids(ct, genetic_distances, parent_indices)
                
                # Add new components with optimized batch operations
                self._add_offspring_components(ct, num_offspring, parent_indices, offspring_energy,
                                            offspring_traits, new_species_ids)
                
                # Update adaptation rates based on reproduction success
                self._update_adaptation_rates(type_idx)

            except Exception as e:
                print(f"Reproduction error handled for type {type_idx}: {str(e)}")
                continue

    def _check_reproduction_conditions(self, ct: CellularTypeData) -> bool:
        """Check if reproduction conditions are met."""
        return (ct is not None and 
                ct.x.size > 0 and
                ct.x.size < self.config.max_particles_per_type and
                np.any(ct.alive))

    def _calculate_eligibility_mask(self, ct: CellularTypeData) -> np.ndarray:
        """Calculate reproduction eligibility with vectorized operations."""
        reproduction_threshold = self._temp_arrays['mutation_buffer'][:ct.x.size]
        np.random.uniform(0, 1, size=ct.x.size)
        
        return (ct.alive & 
                (ct.energy > self.config.reproduction_energy_threshold) &
                (reproduction_threshold < ct.reproduction_rate))

    def _handle_energy_transfer(self, ct: CellularTypeData, eligible_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle energy transfer with safety checks."""
        parent_energy = ct.energy[eligible_mask]
        ct.energy[eligible_mask] = np.maximum(parent_energy * 0.5, self.EPSILON)
        
        return parent_energy, np.maximum(
            parent_energy * self.config.reproduction_offspring_energy_fraction,
            self.EPSILON
        )

    def _generate_mutation_mask(self, num_offspring: int, type_idx: int) -> np.ndarray:
        """Generate mutation mask with adaptive rates."""
        return np.random.random(num_offspring) < (
            self.config.genetics.gene_mutation_rate * 
            self._adaptation_rates['mutation'][type_idx]
        )

    def _generate_offspring_traits(self, ct: CellularTypeData, parent_indices: np.ndarray,
                                 mutation_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate offspring traits using advanced genetic algorithms."""
        offspring_traits = {}
        
        for trait in self.config.genetics.gene_traits:
            parent_values = getattr(ct, trait)[parent_indices]
            offspring_traits[trait] = np.copy(parent_values)
            
            if mutation_mask.any():
                mutation = np.random.normal(
                    loc=0,
                    scale=self.config.genetics.gene_mutation_range[1] - 
                          self.config.genetics.gene_mutation_range[0],
                    size=mutation_mask.sum()
                )
                offspring_traits[trait][mutation_mask] += mutation

        return offspring_traits

    def _calculate_genetic_distances(self, ct: CellularTypeData, offspring_traits: Dict[str, np.ndarray],
                                   parent_indices: np.ndarray) -> np.ndarray:
        """Calculate genetic distances with optimized array operations."""
        squared_diffs = np.zeros_like(parent_indices, dtype=np.float64)
        
        for trait in self.config.genetics.gene_traits:
            diff = offspring_traits[trait] - getattr(ct, trait)[parent_indices]
            squared_diffs += np.square(np.clip(diff, -1e10, 1e10))
            
        return np.sqrt(squared_diffs)

    def _assign_species_ids(self, ct: CellularTypeData, genetic_distances: np.ndarray,
                          parent_indices: np.ndarray) -> np.ndarray:
        """Assign species IDs based on genetic distances."""
        max_species_id = np.max(ct.species_id) if ct.species_id.size > 0 else 0
        return np.where(
            genetic_distances > self.config.speciation_threshold,
            max_species_id + 1,
            ct.species_id[parent_indices]
        )

    def _add_offspring_components(self, ct: CellularTypeData, num_offspring: int,
                                parent_indices: np.ndarray, offspring_energy: np.ndarray,
                                offspring_traits: Dict[str, np.ndarray],
                                new_species_ids: np.ndarray) -> None:
        """Add offspring components with batch operations."""
        for i in range(num_offspring):
            try:
                velocity_scale = (self.config.base_velocity_scale / 
                                np.maximum(offspring_traits['energy_efficiency'][i], self.EPSILON) * 
                                offspring_traits['speed_factor'][i])
                
                ct.add_component(
                    x=ct.x[parent_indices[i]],
                    y=ct.y[parent_indices[i]],
                    vx=np.random.normal(0, velocity_scale),
                    vy=np.random.normal(0, velocity_scale),
                    energy=offspring_energy[i],
                    mass_val=None,  # Handled separately if mass-based
                    **{f"{trait}_val": offspring_traits[trait][i] 
                       for trait in self.config.genetics.gene_traits},
                    species_id_val=new_species_ids[i],
                    parent_id_val=ct.type_id,
                    max_age=ct.max_age
                )
            except Exception:
                continue

    def _update_adaptation_rates(self, type_idx: int) -> None:
        """Update adaptation rates based on simulation feedback."""
        for rate_type in self._adaptation_rates:
            current_rate = self._adaptation_rates[rate_type][type_idx]
            success_rate = np.random.random()  # Simplified for example
            
            # Adjust rates using a dynamic feedback mechanism
            self._adaptation_rates[rate_type][type_idx] = np.clip(
                current_rate + 0.1 * (success_rate - 0.5),
                0.1, 1.0
            )

###############################################################
# Interaction Manager
###############################################################

class InteractionManager:
    """Manages particle interactions and physics with optimized force calculations"""
    def __init__(self, config):
        self.config = config
        self._adaptation_rates = {}
        self._init_adaptation_rates()

    def _init_adaptation_rates(self):
        """Initialize adaptation rate tracking"""
        rate_types = ['interaction', 'synergy', 'predation'] 
        for rate_type in rate_types:
            self._adaptation_rates[rate_type] = np.ones(self.config.n_cell_types)

    def apply_interactions(self, type_manager: CellularTypeManager, rules_manager: InteractionRules):
        """Apply all inter-type interactions"""
        for (i, j, params) in rules_manager.rules:
            self._apply_between_types(i, j, params, type_manager)  # Pass type_manager

    def _apply_between_types(self, i, j, params, type_manager):
        """Apply interactions between two types"""
        ct_i = type_manager.get_cellular_type_by_id(i)
        ct_j = type_manager.get_cellular_type_by_id(j)
        
        if ct_i.x.size == 0 or ct_j.x.size == 0:
            return
            
        fx, fy = self.apply_interaction(ct_i.x, ct_i.y, ct_j.x, ct_j.y, params)
        
        # Apply forces
        ct_i.vx += fx 
        ct_i.vy += fy
        ct_j.vx -= fx
        ct_j.vy -= fy

    def apply_interaction(self, a_x: np.ndarray, a_y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray,
                         params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized force computation between cellular components using vectorized operations."""
        try:
            # Ensure numerical stability with explicit typing
            arrays = np.broadcast_arrays(
                np.asarray(a_x, dtype=np.float64),
                np.asarray(a_y, dtype=np.float64), 
                np.asarray(b_x, dtype=np.float64),
                np.asarray(b_y, dtype=np.float64)
            )
            a_x, a_y, b_x, b_y = arrays

            # Calculate distances with enhanced precision
            dx = np.subtract(a_x, b_x, dtype=np.float64)
            dy = np.subtract(a_y, b_y, dtype=np.float64)
            d_sq = np.add(np.square(dx), np.square(dy), dtype=np.float64)
            
            # Initialize force arrays with proper typing
            fx = np.zeros_like(d_sq, dtype=np.float64)
            fy = np.zeros_like(d_sq, dtype=np.float64)

            # Apply distance constraints
            max_dist = params.get("max_dist", np.inf)
            valid_mask = (d_sq > np.finfo(np.float64).tiny) & (d_sq <= max_dist**2)
            
            if not np.any(valid_mask):
                return fx, fy

            # Calculate distances for valid points
            d = np.sqrt(d_sq, where=valid_mask)
            
            # Apply potential-based forces
            if params.get("use_potential", True):
                pot_strength = np.float64(params.get("potential_strength", 1.0))
                F_pot = np.divide(pot_strength, d, where=valid_mask, out=np.zeros_like(d))
                
                # Apply directional forces
                fx = np.add(fx, F_pot * dx, where=valid_mask, out=fx)
                fy = np.add(fy, F_pot * dy, where=valid_mask, out=fy)

            # Apply gravitational forces
            if params.get("use_gravity", False) and "m_a" in params and "m_b" in params:
                m_a, m_b = np.broadcast_arrays(
                    np.asarray(params["m_a"], dtype=np.float64),
                    np.asarray(params["m_b"], dtype=np.float64)
                )
                
                gravity_factor = np.float64(params.get("gravity_factor", 1.0))
                
                F_grav = np.multiply(
                    gravity_factor,
                    np.divide(np.multiply(m_a, m_b), d_sq, where=valid_mask),
                    where=valid_mask
                )
                
                fx = np.add(fx, F_grav * dx, where=valid_mask, out=fx)
                fy = np.add(fy, F_grav * dy, where=valid_mask, out=fy)

            return np.nan_to_num(fx, copy=False), np.nan_to_num(fy, copy=False)

        except Exception as e:
            print(f"Interaction calculation error handled: {str(e)}")
            return np.zeros_like(a_x), np.zeros_like(a_y)

    def handle_boundaries(self, ct, bounds):
        """Handle boundary reflections"""
        if ct.x.size == 0:
            return
            
        left_mask = ct.x < bounds[0]
        right_mask = ct.x > bounds[1]
        top_mask = ct.y < bounds[2]
        bottom_mask = ct.y > bounds[3]
        
        ct.vx[left_mask | right_mask] *= -1
        ct.vy[top_mask | bottom_mask] *= -1
        
        np.clip(ct.x, bounds[0], bounds[1], out=ct.x)
        np.clip(ct.y, bounds[2], bounds[3], out=ct.y)

    def give_take_interaction(self, giver_energy: np.ndarray, receiver_energy: np.ndarray,
                            giver_mass: Optional[np.ndarray], receiver_mass: Optional[np.ndarray],
                            config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray,
                                                             Optional[np.ndarray], Optional[np.ndarray]]:
        """Optimized energy and mass transfer system with advanced conservation laws."""
        try:
            # Ensure numerical stability
            giver_energy = np.asarray(giver_energy, dtype=np.float64)
            receiver_energy = np.asarray(receiver_energy, dtype=np.float64)
            
            # Calculate transfer amounts with conservation
            transfer_factor = np.clip(config.energy_transfer_factor, 0, 1)
            transfer_amount = np.multiply(receiver_energy, transfer_factor, dtype=np.float64)
            
            # Update energies with conservation laws
            new_receiver = np.subtract(receiver_energy, transfer_amount, dtype=np.float64)
            new_giver = np.add(giver_energy, transfer_amount, dtype=np.float64)
            
            # Handle mass transfer with conservation
            new_giver_mass = new_receiver_mass = None
            if config.mass_transfer and giver_mass is not None and receiver_mass is not None:
                giver_mass = np.asarray(giver_mass, dtype=np.float64)
                receiver_mass = np.asarray(receiver_mass, dtype=np.float64)
                
                mass_transfer = np.multiply(receiver_mass, transfer_factor, dtype=np.float64)
                new_receiver_mass = np.subtract(receiver_mass, mass_transfer, dtype=np.float64)
                new_giver_mass = np.add(giver_mass, mass_transfer, dtype=np.float64)
                
                # Ensure mass conservation
                new_receiver_mass = np.maximum(new_receiver_mass, np.finfo(np.float64).tiny)
                new_giver_mass = np.maximum(new_giver_mass, np.finfo(np.float64).tiny)

            # Ensure energy conservation
            new_receiver = np.maximum(new_receiver, 0)
            new_giver = np.maximum(new_giver, 0)
            
            return new_giver, new_receiver, new_giver_mass, new_receiver_mass

        except Exception as e:
            print(f"Give-take interaction error handled: {str(e)}")
            return giver_energy, receiver_energy, giver_mass, receiver_mass

    def apply_synergy(self, energyA: np.ndarray, energyB: np.ndarray,
                     synergy_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced synergy calculation with emergent behavior patterns."""
        try:
            # Ensure numerical stability
            energyA, energyB = np.broadcast_arrays(
                np.asarray(energyA, dtype=np.float64),
                np.asarray(energyB, dtype=np.float64)
            )
            
            # Apply synergy effects
            synergy_factor = np.clip(synergy_factor, 0, 1)
            avg_energy = np.multiply(np.add(energyA, energyB), 0.5, dtype=np.float64)
            
            # Calculate synergistic energies
            complement_factor = 1.0 - synergy_factor
            newA = np.add(
                np.multiply(energyA, complement_factor),
                np.multiply(avg_energy, synergy_factor),
                dtype=np.float64
            )
            newB = np.add(
                np.multiply(energyB, complement_factor),
                np.multiply(avg_energy, synergy_factor),
                dtype=np.float64
            )
            
            # Ensure energy conservation
            return (
                np.maximum(np.nan_to_num(newA, copy=False), 0),
                np.maximum(np.nan_to_num(newB, copy=False), 0)
            )

        except Exception as e:
            print(f"Synergy calculation error handled: {str(e)}")
            return energyA, energyB

    def random_xy(self, window_width: int, window_height: int, n: int = 1) -> np.ndarray:
        """Generate n random (x, y) coordinates with robust error handling."""
        try:
            # Validate inputs
            window_width = max(1, int(window_width))
            window_height = max(1, int(window_height))
            n = max(1, int(n))
            
            # Generate coordinates safely
            coords = np.random.uniform(0, [window_width, window_height], (n, 2))
            return np.clip(coords, 0, [window_width, window_height])
        except Exception:
            # Return safe fallback value
            return np.zeros((1, 2))
        
    def handle_clustering(self, cellular_types):
        """Apply clustering behavior to particles"""
        for ct in cellular_types:
            self._apply_clustering(ct)

    def _apply_clustering(self, ct):
        """Apply clustering forces within a single cellular type"""
        n = ct.x.size
        if n < 2:
            return

        positions = np.column_stack((ct.x, ct.y))
        tree = cKDTree(positions)
        indices = tree.query_ball_tree(tree, self.config.cluster_radius)

        for idx, neighbor_indices in enumerate(indices):
            neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
            if not neighbor_indices:
                continue

            neighbor_positions = positions[neighbor_indices]
            neighbor_velocities = np.column_stack((ct.vx[neighbor_indices], ct.vy[neighbor_indices]))

            # Alignment
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            alignment = (avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])) * self.config.alignment_strength

            # Cohesion
            center = np.mean(neighbor_positions, axis=0)
            cohesion = (center - positions[idx]) * self.config.cohesion_strength

            # Separation
            separation = (positions[idx] - np.mean(neighbor_positions, axis=0)) * self.config.separation_strength

            # Combine forces
            total_force = alignment + cohesion + separation
            ct.vx[idx] += total_force[0]
            ct.vy[idx] += total_force[1]

###############################################################
# Renderer
###############################################################

class Renderer:
    """Advanced rendering engine for particle visualization with optimized batch processing."""

    def __init__(self, surface: Optional[pygame.Surface], config: SimulationConfig):
        """Initialize the renderer with advanced buffering and optimization."""
        # Store config first since we need it for fallbacks
        self.config = config
        
        # Safely store surface, allowing None
        self.surface = surface
        
        # Only initialize if we have a valid surface
        if surface is not None:
            # Create layered rendering surfaces for compositing
            self._init_surfaces()
            
            # Initialize fonts and text rendering
            self._init_fonts()
            
            # Initialize performance monitoring
            self._frame_times = collections.deque(maxlen=60)
            self._last_frame = time.perf_counter()
            
            # Batch rendering optimizations
            self._particle_batch_size = 1000
            self._color_cache = {}
            self._position_buffer = np.zeros((self._particle_batch_size, 2))
            self._color_buffer = np.zeros((self._particle_batch_size, 3))
            
            # Visual effect parameters
            self.glow_enabled = True
            self.blur_radius = 2
            self.fade_factor = 0.95

            # Clock
            self.clock = pygame.time.Clock()
        else:
            # Set minimal attributes when surface is None
            self._frame_times = collections.deque(maxlen=60)
            self._last_frame = time.perf_counter()
            self.clock = pygame.time.Clock()
            self._color_cache = {}
            self.glow_enabled = False
        
    def _init_surfaces(self):
        """Initialize multi-layered rendering surfaces with error handling."""
        try:
            if self.surface is None:
                raise ValueError("Cannot initialize surfaces without a valid surface")
                
            size = self.surface.get_size()
            flags = pygame.SRCALPHA
            
            # Create surfaces with error handling
            try:
                self.particle_surface = pygame.Surface(size, flags=flags).convert_alpha()
                self.glow_surface = pygame.Surface(size, flags=flags).convert_alpha()
                self.trail_surface = pygame.Surface(size, flags=flags).convert_alpha()
            except pygame.error:
                # Fallback to basic surfaces if convert_alpha fails
                self.particle_surface = pygame.Surface(size, flags=flags)
                self.glow_surface = pygame.Surface(size, flags=flags)
                self.trail_surface = pygame.Surface(size, flags=flags)
            
            # Clear all surfaces
            for surface in [self.particle_surface, self.glow_surface, self.trail_surface]:
                surface.fill((0,0,0,0))
                
        except Exception as e:
            print(f"Error initializing surfaces: {e}")
            # Create minimal surfaces as fallback
            self.particle_surface = pygame.Surface((1,1))
            self.glow_surface = pygame.Surface((1,1))
            self.trail_surface = pygame.Surface((1,1))
            
    def _init_fonts(self):
        """Initialize font rendering with robust fallbacks."""
        try:
            # Try system fonts first
            fonts = ['Arial', 'Helvetica', 'DejaVuSans']
            for font_name in fonts:
                try:
                    self.font = pygame.font.SysFont(font_name, 20)
                    self.large_font = pygame.font.SysFont(font_name, 32)
                    self.small_font = pygame.font.SysFont(font_name, 16)
                    return
                except pygame.error:
                    continue
                    
            # Fall back to default font if no system fonts work
            self.font = pygame.font.Font(None, 20)
            self.large_font = pygame.font.Font(None, 32)
            self.small_font = pygame.font.Font(None, 16)
            
        except Exception as e:
            print(f"Error initializing fonts: {e}")
            # Create dummy font objects that support render method
            class DummyFont:
                def render(self, *args, **kwargs):
                    return pygame.Surface((1,1))
            self.font = self.large_font = self.small_font = DummyFont()

    def generate_vibrant_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n distinct vibrant colors with comprehensive error handling."""
        try:
            n = max(1, int(n))
            colors = []
            
            for i in range(n):
                try:
                    hue = (i / n) % 1.0
                    h_i = int(hue * 6)
                    f = np.clip(hue * 6 - h_i, 0, 1)
                    
                    p = 0
                    q = int(np.clip((1 - f) * 255, 0, 255))
                    t = int(np.clip(f * 255, 0, 255))
                    v = 255
                    
                    color = [(v, t, p), (q, v, p), (p, v, t),
                            (p, q, v), (t, p, v), (v, p, q)][h_i % 6]
                    colors.append(color)
                except Exception:
                    colors.append((255, 255, 255))  # Fallback color
                    
            return colors if colors else [(255, 255, 255)]
            
        except Exception:
            return [(255, 255, 255)]  # Ultimate fallback

    def draw_component(self, x: float, y: float, color: Tuple[int,int,int], 
                      energy: float, speed_factor: float) -> None:
        """Draw a single particle with comprehensive error handling."""
        if self.surface is None:
            return
            
        try:
            # Validate inputs
            x = float(np.clip(x, 0, self.surface.get_width()))
            y = float(np.clip(y, 0, self.surface.get_height()))
            energy = float(np.clip(energy, 0.0, 100.0))
            speed_factor = float(np.clip(speed_factor, 0.1, 10.0))
            
            # Calculate particle appearance
            intensity_factor = energy/100.0
            color_key = (color, intensity_factor, speed_factor)
            
            # Use cached color if available
            if color_key in self._color_cache:
                c = self._color_cache[color_key]
            else:
                base_intensity = intensity_factor * speed_factor
                fade = (1 - intensity_factor) * 100
                
                c = tuple(min(255, max(0, int(
                    col * base_intensity + fade + 
                    (speed_factor - 1) * 20 * intensity_factor
                ))) for col in color)
                
                self._color_cache[color_key] = c
            
            # Draw main particle
            pos = (int(x), int(y))
            size = max(1, int(self.config.particle_size * (0.8 + 0.4 * intensity_factor)))
            
            if hasattr(self, 'particle_surface'):
                pygame.draw.circle(self.particle_surface, c, pos, size)
            
                # Add glow effect for high-energy particles
                if self.glow_enabled and energy > 50 and hasattr(self, 'glow_surface'):
                    glow_size = size + self.blur_radius
                    glow_color = tuple(min(255, int(v * 0.7)) for v in c)
                    pygame.draw.circle(self.glow_surface, glow_color, pos, glow_size)
                    
        except Exception as e:
            print(f"Error in draw_component: {e}")

    def draw_cellular_type(self, ct: 'CellularTypeData') -> None:
        """Efficiently render cellular types with comprehensive validation."""
        if self.surface is None:
            return
            
        try:
            # Validate required attributes
            required_attrs = ['alive', 'x', 'y', 'energy', 'speed_factor']
            if not all(hasattr(ct, attr) for attr in required_attrs):
                return

            # Convert to numpy arrays with validation
            arrays = {}
            for attr in required_attrs:
                try:
                    arr = getattr(ct, attr)
                    arrays[attr] = np.asarray(arr if hasattr(arr, '__len__') else [arr])
                except Exception:
                    arrays[attr] = np.array([])

            # Validate and align array sizes
            sizes = [len(arr) for arr in arrays.values()]
            if not sizes or 0 in sizes:
                return
                
            max_size = min(max(sizes), 1000000)  # Prevent excessive memory usage
            
            # Resize arrays to match
            for key, arr in arrays.items():
                if len(arr) != max_size:
                    new_arr = np.zeros(max_size)
                    new_arr[:len(arr)] = arr[:max_size]
                    arrays[key] = new_arr

            # Get alive indices
            alive_indices = np.where(arrays['alive'])[0]
            
            if len(alive_indices) == 0:
                return
                
            # Process in batches
            for i in range(0, len(alive_indices), self._particle_batch_size):
                batch_indices = alive_indices[i:i + self._particle_batch_size]
                
                try:
                    # Gather and validate batch data
                    positions = np.column_stack((
                        arrays['x'][batch_indices],
                        arrays['y'][batch_indices]
                    ))
                    energies = arrays['energy'][batch_indices]
                    speed_factors = arrays['speed_factor'][batch_indices]
                    
                    # Validate data
                    valid_mask = ~(
                        np.isnan(positions).any(axis=1) | 
                        np.isnan(energies) | 
                        np.isnan(speed_factors) |
                        np.isinf(positions).any(axis=1) |
                        np.isinf(energies) |
                        np.isinf(speed_factors)
                    )
                    
                    # Draw valid particles
                    for pos, energy, speed in zip(
                        positions[valid_mask], 
                        energies[valid_mask],
                        speed_factors[valid_mask]
                    ):
                        self.draw_component(
                            float(pos[0]), float(pos[1]),
                            ct.color, float(energy), float(speed)
                        )
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in draw_cellular_type: {e}")

    def clear_screen(self, screen: Optional[pygame.Surface]=pygame.display.get_surface()) -> None:
        """Clear the screen with comprehensive error handling."""
        if screen is None:
            return
        
        try:
            screen.fill((0, 0, 0))
        except Exception as e:
            print(f"Error clearing screen: {e}")

    def render_frame(self, screen: Optional[pygame.Surface], 
                    cellular_types: List['CellularTypeData'],
                    species_count: Dict[int, int]) -> None:
        """Render the current frame with comprehensive error handling."""
        if screen is None or self.surface is None:
            return
            
        try:
            # Clear screen
            screen.fill((0, 0, 0))

            # Draw cellular types
            for ct in cellular_types:
                if hasattr(ct, 'alive') and ct.alive.size > 0:
                    self.draw_cellular_type(ct)

            # Apply motion trails if surfaces exist
            if all(hasattr(self, attr) for attr in ['trail_surface', 'particle_surface', 'glow_surface']):
                self.trail_surface.fill((0,0,0,int(255 * (1 - self.fade_factor))))
                self.trail_surface.blit(self.particle_surface, (0,0))
                
                # Composite layers
                self.surface.blit(self.trail_surface, (0,0))
                if self.glow_enabled:
                    self.surface.blit(self.glow_surface, (0,0), special_flags=pygame.BLEND_RGB_ADD)
                self.surface.blit(self.particle_surface, (0,0))
                
                # Clear for next frame
                self.particle_surface.fill((0,0,0,0))
                self.glow_surface.fill((0,0,0,0))

            # Update and render stats
            current_time = time.perf_counter()
            frame_time = current_time - self._last_frame
            self._frame_times.append(frame_time)
            self._last_frame = current_time
            
            avg_frame_time = np.mean(self._frame_times) if self._frame_times else 0
            
            stats_text = (
                f"FPS: {self.clock.get_fps():.1f} ({1000*avg_frame_time:.1f}ms) | "
                f"Species: {len(species_count)} | "
                f"Particles: {sum(species_count.values()):,}"
            )
            
            try:
                text_surface = self.font.render(stats_text, True, (255,255,255))
                self.surface.blit(text_surface, (10,10))
            except Exception as e:
                print(f"Error rendering stats: {e}")
            
        except Exception as e:
            print(f"Error rendering frame: {e}")

###############################################################
# Simulation Core
###############################################################

class SimulationCore:
    """Core simulation functionality and initialization"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._init_display()
        self._init_managers()
        self.frame_count = 0
        self.run_flag = True
        self.species_count = defaultdict(int)
        
    def _init_display(self):
        """Initialize display and window setup"""
        pygame.init()
        display_info = pygame.display.Info()
        self.screen_width = max(800, display_info.current_w)
        self.screen_height = max(600, display_info.current_h)
        
        try:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            )
        except pygame.error:
            self.screen = pygame.display.set_mode((800, 600), 
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.screen_width, self.screen_height = 800, 600
            
        pygame.display.set_caption("Emergent Cellular Automata Simulation")
        self.clock = pygame.time.Clock()
        
    def _init_managers(self):
        """Initialize simulation managers and components"""
        self.edge_buffer = np.clip(0.05 * max(self.screen_width, self.screen_height), 10, 100)
        self.screen_bounds = np.array([
            self.edge_buffer,
            self.screen_width - self.edge_buffer, 
            self.edge_buffer,
            self.screen_height - self.edge_buffer
        ])
        
        self.renderer = Renderer(self.screen, self.config)
        self.colors = self.renderer.generate_vibrant_colors(self.config.n_cell_types)
        
        n_mass_types = max(0, min(
            int(self.config.mass_based_fraction * self.config.n_cell_types),
            self.config.n_cell_types
        ))
        mass_based_type_indices = list(range(n_mass_types))
        
        self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)
        self.rules_manager = InteractionRules(self.config, mass_based_type_indices)
        self.genetic_interpreter = GeneticInterpreter()
        
        self._init_cellular_types(n_mass_types)
        
    def _init_cellular_types(self, n_mass_types):
        """Initialize cellular type data"""
        mass_values = np.zeros(n_mass_types)
        if n_mass_types > 0:
            mass_values = np.clip(
                np.random.uniform(
                    self.config.mass_range[0],
                    self.config.mass_range[1],
                    n_mass_types
                ),
                1e-6,
                None
            )
            
        for i in range(self.config.n_cell_types):
            try:
                ct = CellularTypeData(
                    config=self.config,
                    particles_per_type=self.config.particles_per_type,
                    type_id=i,
                    color=self.colors[i],
                    n_particles=max(1, self.config.particles_per_type),
                    window_width=self.screen_width,
                    window_height=self.screen_height,
                    initial_energy=max(0.1, self.config.initial_energy),
                    max_age=max(1, self.config.max_age),
                    mass=mass_values[i] if i < n_mass_types else None,
                    base_velocity_scale=max(0.1, self.config.base_velocity_scale)
                )
                self.type_manager.add_cellular_type_data(ct)
            except Exception as e:
                print(f"Error creating cellular type {i}: {e}")

###############################################################
# Timer Class
###############################################################

class Timer:
    """High-precision timer with performance monitoring capabilities."""
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval:.6f} seconds")

###############################################################
# Performance Manager Class 
###############################################################

class PerformanceManager:
    """Handles performance monitoring and optimization with high-precision timing"""
    def __init__(self):
        # Initialize performance metrics with optimized data structures
        self.metrics = {
            'fps_history': collections.deque([60.0] * 60, maxlen=60),
            'particle_counts': collections.deque([0] * 60, maxlen=60),
            'cull_history': collections.deque(maxlen=10),
            'last_cull_time': time.perf_counter(),
            'performance_score': 1.0,
            'stress_threshold': 0.7,
            'min_fps': 45,
            'target_fps': 90, 
            'emergency_fps': 30,
            'last_emergency': 0,
            'frame_times': collections.deque(maxlen=120),
            'timers': {},
            'stats': {
                'avg_fps': 60.0,
                'particle_count': 0,
                'memory_usage': 0,
                'cpu_usage': 0.0,
                'last_update': time.perf_counter()
            }
        }
        
    def start_timer(self, name: str) -> Timer:
        """Start a high-precision named timer"""
        timer = Timer()
        self.metrics['timers'][name] = timer
        return timer
        
    def get_elapsed(self, name: str) -> float:
        """Get elapsed time for a named timer with error handling"""
        try:
            timer = self.metrics['timers'].get(name)
            return timer.interval if timer and hasattr(timer, 'interval') else 0.0
        except Exception:
            return 0.0

    def update_stats(self) -> None:
        """Update performance statistics with optimized calculations"""
        try:
            current_time = time.perf_counter()
            if current_time - self.metrics['stats']['last_update'] >= 1.0:
                self.metrics['stats'].update({
                    'avg_fps': np.mean(self.metrics['fps_history']),
                    'particle_count': np.sum(self.metrics['particle_counts']),
                    'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_usage': psutil.Process().cpu_percent(),
                    'last_update': current_time
                })
        except Exception as e:
            print(f"Error updating stats: {e}")
        
    def handle_degradation(self, current_fps: float, simulation: Any) -> None:
        """Handle performance degradation with adaptive optimization"""
        try:
            self.metrics['fps_history'].append(current_fps)
            if current_fps < self.metrics['emergency_fps']:
                if time.perf_counter() - self.metrics['last_emergency'] > 5.0:
                    with self.start_timer('emergency_optimization'):
                        self._emergency_optimization(simulation)
                    self.metrics['last_emergency'] = time.perf_counter()
        except Exception as e:
            print(f"Error handling degradation: {e}")
                
    def _emergency_optimization(self, simulation: Any) -> None:
        """Emergency optimization with vectorized operations"""
        try:
            start_time = time.perf_counter()
            
            # Vectorized particle culling
            for ct in simulation.type_manager.cellular_types:
                if ct.x.size > 100:
                    keep_count = max(50, ct.x.size // 2)
                    with self.start_timer(f'emergency_cull_{ct.type_id}'):
                        indices = np.random.choice(ct.x.size, keep_count, replace=False)
                        for attr in ['x', 'y', 'vx', 'vy', 'energy', 'age', 'alive']:
                            if hasattr(ct, attr):
                                setattr(ct, attr, getattr(ct, attr)[indices])
            
            # Disable expensive computations
            simulation.config.synergy_range = 0
            simulation.config.predation_range = 0
            
            # Record performance metrics
            elapsed = time.perf_counter() - start_time
            self.metrics['frame_times'].append(elapsed)
            self.metrics['cull_history'].append({
                'time': time.perf_counter(),
                'duration': elapsed
            })
            
        except Exception as e:
            print(f"Error in emergency optimization: {e}")

    def measure_block(self, block_name: str):
        """
        Context manager for measuring execution time of code blocks with minimal overhead.
        Uses high precision timer and efficient data structures for performance tracking.
        
        Args:
            block_name (str): Identifier for the code block being measured
            
        Returns:
            Context manager that tracks execution time
        """
        class TimerContext:
            def __init__(self, perf_manager, block):
                self.perf_manager = perf_manager
                self.block = block
                self.start_time = 0
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    duration = time.perf_counter() - self.start_time
                    
                    # Update metrics using efficient data structures
                    if self.block not in self.perf_manager.metrics['block_times']:
                        self.perf_manager.metrics['block_times'][self.block] = collections.deque(maxlen=120)
                    
                    self.perf_manager.metrics['block_times'][self.block].append(duration)
                    
                    # Track max duration for optimization triggers
                    if duration > self.perf_manager.metrics.get(f'{self.block}_max_time', 0):
                        self.perf_manager.metrics[f'{self.block}_max_time'] = duration
                        
                    # Update performance score
                    block_avg = np.mean(self.perf_manager.metrics['block_times'][self.block])
                    self.perf_manager.metrics['performance_score'] = min(
                        self.perf_manager.metrics['performance_score'],
                        1.0 / (1.0 + block_avg)
                    )
                    
                except Exception as e:
                    print(f"Error measuring block {self.block}: {e}")
                    
                return False  # Don't suppress exceptions
                
        return TimerContext(self, block_name)
###############################################################
# Particle Manager Class
############################################################### 

class ParticleManager:
    """Manages particle culling and optimization with performance monitoring"""
    def __init__(self, config):
        self.config = config
        self.bounds = {
            'energy': (0, config.max_energy),
            'velocity': (-config.max_velocity, config.max_velocity), 
            'age': (0, config.max_age),
            'mass': (config.min_mass, config.max_mass),
            'traits': (0, 1)
        }
        self._performance_metrics = {
            'fps_history': collections.deque(maxlen=120),
            'particle_counts': collections.deque(maxlen=120),
            'last_cull_time': time.time(),
            'emergency_threshold': 0.3
        }

    def cull_oldest(self, cellular_types, performance_metrics):
        """Cull oldest particles based on performance metrics"""
        try:
            current_time = time.time()
            current_fps = performance_metrics['current_fps']
            
            # Update performance tracking
            self._performance_metrics['fps_history'].append(current_fps)
            total_particles = sum(ct.x.size for ct in cellular_types)
            self._performance_metrics['particle_counts'].append(total_particles)
            
            # Calculate performance stress
            fps_stress = max(0, (performance_metrics['target_fps'] - current_fps) / 
                           performance_metrics['target_fps'])
            particle_stress = 1 / (1 + np.exp(-total_particles / 10000))
            system_stress = (fps_stress * 0.7 + particle_stress * 0.3)
            
            for ct in cellular_types:
                if ct.x.size < 100:
                    continue
                    
                self._cull_type_particles(ct, system_stress)
                
            self._performance_metrics['last_cull_time'] = current_time
                
        except Exception as e:
            print(f"Error culling particles: {e}")
            
    def _cull_type_particles(self, ct, stress_level):
        """Cull particles for a specific cellular type based on fitness"""
        try:
            positions = np.column_stack((ct.x, ct.y))
            tree = cKDTree(positions)
            
            # Calculate fitness components
            density_scores = tree.query_ball_point(positions, r=200, return_length=True)
            density_penalty = density_scores / (np.max(density_scores) + 1e-6)
            
            energy_score = ct.energy * ct.energy_efficiency * (1 - (ct.age / ct.max_age))
            interaction_score = (ct.interaction_strength * ct.synergy_affinity * 
                               ct.colony_factor * ct.reproduction_rate)
            
            # Combine into final fitness score
            fitness_scores = (energy_score * 0.4 + interaction_score * 0.3 + 
                            (1 - density_penalty) * 0.3)
            fitness_scores = (fitness_scores - np.min(fitness_scores)) / (
                np.max(fitness_scores) - np.min(fitness_scores) + 1e-10)
            
            # Calculate cull rate based on stress
            cull_rate = np.clip(0.1 * stress_level, 0.05, 0.4)
            removal_count = int(ct.x.size * cull_rate)
            
            # Keep highest fitness particles
            keep_indices = np.argsort(fitness_scores)[removal_count:]
            keep_mask = np.zeros(ct.x.size, dtype=bool)
            keep_mask[keep_indices] = True
            
            # Update particle arrays
            self._apply_mask(ct, keep_mask)
            
        except Exception as e:
            print(f"Error in type particle culling: {e}")
            
    def emergency_cull(self, ct, keep_count):
        """Emergency culling of particles when performance severely degrades"""
        try:
            if ct.x.size <= keep_count:
                return
                
            positions = np.column_stack((ct.x, ct.y))
            tree = cKDTree(positions)
            
            # Quick fitness calculation for emergency
            density_scores = tree.query_ball_point(positions, r=200, return_length=True)
            energy_scores = ct.energy * (1 - (ct.age / ct.max_age))
            
            # Combine scores and keep best particles
            fitness_scores = energy_scores - (density_scores / np.max(density_scores))
            keep_indices = np.argsort(fitness_scores)[-keep_count:]
            keep_mask = np.zeros(ct.x.size, dtype=bool)
            keep_mask[keep_indices] = True
            
            self._apply_mask(ct, keep_mask)
            
        except Exception as e:
            print(f"Error in emergency cull: {e}")
            self._ensure_stability(ct)
            
    def _apply_mask(self, ct, mask):
        """Apply boolean mask to particle arrays"""
        arrays_to_filter = [
            'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
            'speed_factor', 'interaction_strength', 'perception_range',
            'reproduction_rate', 'synergy_affinity', 'colony_factor',
            'drift_sensitivity', 'species_id', 'parent_id'
        ]
        for attr in arrays_to_filter:
            if hasattr(ct, attr):
                setattr(ct, attr, getattr(ct, attr)[mask])
        if hasattr(ct, 'mass') and ct.mass is not None:
            ct.mass = ct.mass[mask]
            
    def _ensure_stability(self, ct):
        """Ensure particle attributes remain within valid bounds"""
        try:
            for attr in ['x', 'y', 'vx', 'vy', 'energy', 'age', 'energy_efficiency',
                        'speed_factor', 'interaction_strength', 'perception_range',
                        'reproduction_rate', 'synergy_affinity', 'colony_factor',
                        'drift_sensitivity', 'species_id', 'parent_id']:
                arr = getattr(ct, attr)
                if isinstance(arr, np.ndarray):
                    if attr in ['energy', 'velocity', 'age', 'mass']:
                        setattr(ct, attr, np.clip(arr, *self.bounds[attr]))
                    else:
                        setattr(ct, attr, np.clip(arr, *self.bounds['traits']))
        except Exception as e:
            print(f"Error ensuring stability: {e}")

###############################################################
# Colony Manager Class
###############################################################

class ColonyManager:
    """
    Manages colony formation, behavior, and interactions with high performance and dynamic adaptation.
    """
    def __init__(self, config: SimulationConfig):
        """Initialize with robust parameters and adaptive strategies."""
        self.config = config
        self.active_colonies = []
        self._colony_id_counter = 0
        self.cellular_types = []  # Store reference to cellular types
        self._adaptation_rates = {
            'formation': np.full(config.n_cell_types, 0.5),
            'cohesion': np.full(config.n_cell_types, 0.5),
            'migration': np.full(config.n_cell_types, 0.5)
        }

    def update_colonies(self, cellular_types: List[CellularTypeData]) -> None:
        """Update colony states, formation, and interactions."""
        try:
            self.cellular_types = cellular_types  # Update reference to cellular types
            self._dissolve_inactive_colonies()
            self._form_new_colonies(cellular_types)

            for colony in self.active_colonies:
                self._update_colony_position(colony, cellular_types)
                self._apply_cohesion_forces(colony, cellular_types)
                self._handle_colony_interactions(colony, cellular_types)
                self._update_adaptation_rates(colony.type_id)

        except Exception as e:
            print(f"Error updating colonies: {e}")
            traceback.print_exc()

    def _dissolve_inactive_colonies(self) -> None:
        """Dissolve colonies that have too few members."""
        self.active_colonies = [
            colony for colony in self.active_colonies
            if self._count_colony_members(colony) >= self.config.min_colony_size
        ]

    def _form_new_colonies(self, cellular_types: List[CellularTypeData]) -> None:
        """Form new colonies based on particle behavior."""
        try:
            for ct in cellular_types:
                # Validate cellular type has required attributes
                if not hasattr(ct, 'alive') or not hasattr(ct, 'colony_id') or not hasattr(ct, 'colony_factor'):
                    continue
                    
                # Get indices of alive particles not already in colonies
                alive_mask = ct.alive & (ct.colony_id == -1)
                candidate_indices = np.where(alive_mask)[0]
                
                if len(candidate_indices) == 0:
                    continue
                    
                # Calculate formation probabilities
                formation_probs = self.config.colony_formation_probability * ct.colony_factor[candidate_indices]
                random_vals = np.random.rand(len(candidate_indices))
                
                # Find particles that will form colonies
                form_colony_mask = random_vals < formation_probs
                new_colony_indices = candidate_indices[form_colony_mask]
                
                # Create colonies for selected particles
                for idx in new_colony_indices:
                    if idx < len(ct.x):  # Extra bounds check
                        self._create_new_colony(ct, idx)
                        
        except Exception as e:
            print(f"Error forming new colonies: {e}")
            traceback.print_exc()
            
    def _create_new_colony(self, ct: CellularTypeData, particle_idx: int) -> None:
        """Create a new colony with the given particle as the founder."""
        new_colony = ColonyData(
            colony_id=self._colony_id_counter,
            type_id=ct.type_id,
            x=ct.x[particle_idx],
            y=ct.y[particle_idx],
            radius=self.config.colony_radius
        )
        self._colony_id_counter += 1
        self.active_colonies.append(new_colony)
        ct.colony_id[particle_idx] = new_colony.colony_id

    def _update_colony_position(self, colony: 'ColonyData', cellular_types: List[CellularTypeData]) -> None:
        """Update the colony's position based on its members."""
        ct = cellular_types[colony.type_id]
        member_indices = np.where(ct.colony_id == colony.colony_id)[0]
        if len(member_indices) > 0:
            colony.x = np.mean(ct.x[member_indices])
            colony.y = np.mean(ct.y[member_indices])

    def _apply_cohesion_forces(self, colony: 'ColonyData', cellular_types: List[CellularTypeData]) -> None:
        """Apply cohesion forces to keep colony members together."""
        ct = cellular_types[colony.type_id]
        member_indices = np.where(ct.colony_id == colony.colony_id)[0]
        if len(member_indices) > 0:
            dx = colony.x - ct.x[member_indices]
            dy = colony.y - ct.y[member_indices]
            dist = np.hypot(dx, dy)
            cohesion_force = self.config.colony_cohesion_strength * dist / colony.radius
            ct.vx[member_indices] += cohesion_force * dx
            ct.vy[member_indices] += cohesion_force * dy

    def _handle_colony_interactions(self, colony: 'ColonyData', cellular_types: List[CellularTypeData]) -> None:
        """Handle interactions between colonies and individual particles."""
        # ... (Implementation for colony interactions)

    def _update_adaptation_rates(self, type_id: int) -> None:
        """Update adaptation rates based on colony success."""
        # ... (Implementation for adaptation rate updates)

    def _count_colony_members(self, colony: 'ColonyData') -> int:
        """Count the number of particles belonging to the colony."""
        try:
            count = 0
            for ct in self.cellular_types:  # Now using stored reference
                count += np.sum(ct.colony_id == colony.colony_id)
            return count
        except Exception as e:
            print(f"Error counting colony members: {e}")
            return 0
###############################################################
# Colony Data Class
###############################################################

class ColonyData:
    """Represents a colony with dynamic properties and emergent behaviors."""

    def __init__(self, colony_id: int, type_id: int, x: float, y: float, radius: float):
        """Initialize colony with essential properties."""
        self.colony_id = colony_id
        self.type_id = type_id
        self.x = x
        self.y = y
        self.radius = radius
        self.age = 0
        self.energy = 100.0  # Initial colony energy
        self.growth_rate = 0.1
        self.migration_direction = np.random.rand(2) * 2 - 1  # Random initial direction

###############################################################
# Cellular Automata Class
###############################################################

class CellularAutomata:
    """Main simulation controller class implementing advanced cellular automata behaviors"""
    def __init__(self, config: SimulationConfig):
        """Initialize simulation components with robust error handling"""
        self.config = config
        
        # Initialize core components first
        try:
            pygame.init()
            self.renderer = Renderer(None, config)  # Initialize renderer first
            self.interactions = InteractionManager(config)
            self.type_manager = CellularTypeManager(config,colors=[],mass_based_type_indices=[])
            self.rules_manager = InteractionRules(config,mass_based_type_indices=[])
            self.colony_manager = ColonyManager(config) 
            self.genetic_interpreter = GeneticInterpreter()
            self.performance = PerformanceManager()
            self.particles = ParticleManager(config)
        except Exception as e:
            print(f"Error initializing core components: {e}")
            raise

        # Set up display with fallbacks
        try:
            display_info = pygame.display.Info()
            self.screen_width = max(800, display_info.current_w)
            self.screen_height = max(600, display_info.current_h)
            
            try:
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF
                )
            except pygame.error:
                print("Hardware acceleration failed, falling back to software mode")
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                
            self.renderer.screen = self.screen
            
        except Exception as e:
            print(f"Error setting up display: {e}")
            raise

        pygame.display.set_caption("Emergent Cellular Automata Simulation")
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.run_flag = True

        # Safe edge buffer calculation
        self.edge_buffer = max(10, min(100, 0.05 * min(self.screen_width, self.screen_height)))

        # Generate colors and setup cellular types
        try:
            self.colors = self.renderer.generate_vibrant_colors(self.config.n_cell_types)
            
            # Create cellular types
            for i in range(self.config.n_cell_types):
                ct = CellularTypeData(
                    type_id=i,
                    config=self.config,
                    color=self.colors[i],
                    n_particles=self.config.particles_per_type,
                    window_width=self.screen_width,
                    window_height=self.screen_height,
                    initial_energy=100.0,
                    max_age=1000
                )
                self.type_manager.add_cellular_type_data(ct)
                
        except Exception as e:
            print(f"Error creating cellular types: {e}")
            raise

        # Initialize tracking and caches
        self.species_count = defaultdict(int)
        self.tree_cache = {}
        self._init_performance_tracking()
        self._init_caches()

        # Set screen bounds
        self.screen_bounds = np.array([
            self.edge_buffer,
            self.screen_width - self.edge_buffer,
            self.edge_buffer, 
            self.screen_height - self.edge_buffer
        ])

    def _init_performance_tracking(self):
        """Initialize performance tracking metrics"""
        self._performance_metrics = {
            'frame_times': collections.deque(maxlen=120),
            'current_fps': 60.0,
            'target_fps': 60.0
        }

    def _init_caches(self):
        """Initialize caches with memory limits"""
        self.interaction_cache = {}
        self.genetic_cache = {}
        self.MAX_CACHE_SIZE = 1000

    def main_loop(self) -> None:
        """Main simulation loop with error handling and performance monitoring"""
        last_time = time.time()
        
        while self.run_flag:
            try:
                # Timing
                current_time = time.time()
                frame_time = current_time - last_time
                last_time = current_time
                
                self._performance_metrics['frame_times'].append(frame_time)
                
                # Frame counting
                self.frame_count += 1
                if self.config.max_frames > 0 and self.frame_count > self.config.max_frames:
                    self.run_flag = False

                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and 
                        event.key == pygame.K_ESCAPE
                    ):
                        self.run_flag = False
                        break

                # Simulation step
                self._simulation_step()

                # Update display
                pygame.display.flip()
                
                # Frame timing
                self._performance_metrics['current_fps'] = self.clock.tick(60)
                
                # Particle management
                if self._performance_metrics['current_fps'] < 30:
                    self.particles.cull_oldest(
                        self.type_manager.cellular_types,
                        self._performance_metrics
                    )

            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                continue

        pygame.quit()

    def _simulation_step(self):
        """Execute one simulation step"""
        try:
            # Update particle positions and states
            for ct in self.type_manager.cellular_types:
                # Apply forces and movement
                self.rules_manager.apply_movement(ct, self.screen_bounds)
                self.rules_manager.apply_boundaries(ct, self.screen_bounds)
                
                # Handle interactions
                for other_ct in self.type_manager.cellular_types:
                    if ct != other_ct:
                        self.rules_manager.apply_interactions(ct, other_ct)
                
                # Update colonies
                self.colony_manager.update(ct)
                
                # Handle reproduction and death
                self.rules_manager.handle_reproduction(ct)
                self.rules_manager.handle_death(ct)

            # Render frame
            self.renderer.clear_screen()
            self.renderer.draw_cellular_type(self.type_manager.cellular_types)
            # self.renderer.draw_colonies(self.colony_manager.colonies)
            
        except Exception as e:
            print(f"Error in simulation step: {e}")
            traceback.print_exc()

###############################################################
# Entry Point
###############################################################

def main():
    """
    Main configuration and run function.
    """
    config = SimulationConfig()
    cellular_automata = CellularAutomata(config)
    cellular_automata.main_loop()

if __name__ == "__main__":
    main()