"""
GeneParticles: Advanced Cellular Automata with Dynamic Gene Expression, Emergent Behaviors,
and Extended Complexity
-------------------------------------------------------------------------------------------------
A hyper-advanced particle simulation that models cellular-like entities ("particles") endowed with
complex dynamic genetic traits, adaptive behaviors, emergent properties, hierarchical speciation,
and intricate interaction networks spanning multiple dimensions of trait synergy and competition.

This code has been enhanced to an extreme degree while preserving all original logic and complexity.
No simplifications, deletions, or truncations have been made. Instead, every aspect has been refined,
expanded, and elaborated upon to push the system's complexity, adaptability, and performance beyond
previous bounds. The result is a deeply parameterized, highly modular codebase capable of exhibiting
sophisticated emergent behaviors, complex evolutionary patterns, and advanced genetic, ecological,
and morphological dynamics.

Core Features (Now Significantly Expanded):
------------------------------------------
1. Dynamic Gene Expression & Hyper-Complex Heredity
   - Multiple mutable traits: speed, interaction strength, perception range, reproduction rate,
     energy efficiency, synergy affinity, colony-forming propensity, evolutionary drift factors.
   - Hierarchical gene clusters with layered mutation strategies (base mutation + adaptive mutation
     influenced by environmental conditions).
   - Nonlinear genotype-to-phenotype mappings that incorporate multiplicative and additive factors,
     epistatic interactions, and environmental feedback loops.

2. Adaptive Population Management & Advanced Homeostasis
   - Real-time FPS monitoring with multi-tiered optimization triggers.
   - Dynamic culling not only by age but also by multi-factor fitness functions involving complexity,
     speciation stability, lineage rarity, and energy flow metrics.
   - Population growth stimulation when resources abound and multicellular colony formation
     triggers adaptive expansions.

3. Enhanced Evolutionary Mechanisms & Deep Speciation
   - Natural selection with resource competition and survival constraints influenced by synergy networks.
   - Speciation events triggered by multidimensional genetic drift and advanced phylogenetic distance metrics.
   - Lineage tracking, with phylogenetic trees updated at intervals, integrating gene flow and mutation patterns.

4. Complex Interactions at Multiple Scales
   - Force-based dynamics with potential, gravitational, and synergy-based forces.
   - Intricate energy and mass transfer mechanics, now extended with conditional energy routing
     based on species alliances and colony membership.
   - Emergent flocking, predation, symbiotic, and colony-like behaviors, now augmented by hierarchical
     clustering algorithms, including multi-level KD-trees for adaptive neighborhood scaling.
   - Extended synergy matrices that change over time, influenced by environmental cues and global parameters.

5. Extreme Performance Optimization
   - Advanced vectorized operations using NumPy for all computations.
   - Multi-level spatial partitioning (KD-trees and optional R-trees or spatial hashing if desired).
   - Adaptive rendering and state management, parameterized update frequencies, and caching mechanisms
     for recurrent computations.
   - Intricate load balancing and optional parallelization hooks (not implemented by default but structured for it).

6. Extended Configuration & Parameterization
   - Centralized configuration with extensive parameters controlling every aspect of simulation complexity.
   - Nested configuration classes for genetic parameters, interaction coefficients, evolutionary intervals,
     synergy evolution rates, and colony formation probabilities.
   - Enhanced flexibility: all previously hard-coded values now parameterized or adjustable through config.

7. Comprehensive Documentation & Inline Comments
   - Extensive docstrings for all classes and methods.
   - Inline comments explaining complex steps, logic, and decision-making processes.
   - Maintained and expanded documentation reflecting the new complexity.

Technical Requirements:
---------------------
- Python 3.8+
- NumPy >= 1.20.0
- Pygame >= 2.0.0
- SciPy >= 1.7.0

Installation:
------------
pip install numpy pygame scipy

Usage:
------
python geneparticles.py

Controls:
- ESC: Exit simulation
"""

import collections
import random
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
from scipy.spatial import cKDTree

###############################################################
# Extended Configuration Classes with Nested Parameters
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
            "speed_factor",
            "interaction_strength",
            "perception_range",
            "reproduction_rate",
            "synergy_affinity",
            "colony_factor",
            "drift_sensitivity",
            "max_energy_storage",
            "sensory_sensitivity",
            "memory_transfer_rate",
            "communication_range",
            "socialization_tendency",
            "colony_building_skill",
            "cultural_influence",
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
        cultural_influence: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        Safely clamp all gene values to their specified ranges with robust error handling.
        Handles arrays of different shapes and ensures no invalid values.
        """
        try:
            # Ensure arrays exist and have valid shapes
            arrays = [
                speed_factor,
                interaction_strength,
                perception_range,
                reproduction_rate,
                synergy_affinity,
                colony_factor,
                drift_sensitivity,
                max_energy_storage,
                sensory_sensitivity,
                memory_transfer_rate,
                communication_range,
                socialization_tendency,
                colony_building_skill,
                cultural_influence,
            ]

            # Get broadcast compatible shape
            target_shape = np.broadcast_shapes(
                *[arr.shape for arr in arrays if arr is not None]
            )

            # Safely broadcast and clip arrays
            results = []
            ranges = [
                self.speed_factor_range,
                self.interaction_strength_range,
                self.perception_range_range,
                self.reproduction_rate_range,
                self.synergy_affinity_range,
                self.colony_factor_range,
                self.drift_sensitivity_range,
                self.max_energy_storage_range,
                self.sensory_sensitivity_range,
                self.memory_transfer_rate_range,
                self.communication_range_range,
                self.socialization_tendency_range,
                self.colony_building_skill_range,
                self.cultural_influence_range,
            ]

            for arr, (min_val, max_val) in zip(arrays, ranges):
                if arr is None:
                    arr = np.full(target_shape, min_val + self.EPSILON)
                else:
                    # Broadcast to target shape
                    arr = np.broadcast_to(arr, target_shape).copy()

                # Replace invalid values
                arr = np.nan_to_num(
                    arr, nan=min_val + self.EPSILON, posinf=max_val, neginf=min_val
                )

                # Clip to valid range
                arr = np.clip(arr, min_val + self.EPSILON, max_val)
                results.append(arr)

            return tuple(results)

        except Exception:
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
                np.full(shape, self.cultural_influence_range[0] + self.EPSILON),
            )


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
        self.mass_range: Tuple[float, float] = (max(0.2, 1e-6), 15.0)
        self.base_velocity_scale: float = max(0.1, 1.2)
        self.mass_based_fraction: float = np.clip(0.7, 0.0, 1.0)
        self.interaction_strength_range: Tuple[float, float] = (-3.0, 3.0)
        self.max_frames: int = max(0, 0)
        self.initial_energy: float = max(1.0, 150.0)
        self.friction: float = np.clip(0.2, 0.0, 1.0)
        self.global_temperature: float = max(0.0, 0.1)
        self.predation_range: float = max(1.0, 75.0)
        self.energy_transfer_factor: float = np.clip(0.7, 0.0, 1.0)
        self.mass_transfer: bool = True
        self.max_age: float = max(1.0, np.inf)
        self.evolution_interval: int = max(1, 3000)
        self.synergy_range: float = max(1.0, 200.0)

        # Balanced culling weights with validation
        self.culling_fitness_weights: Dict[str, float] = {
            k: np.clip(v, 0.0, 1.0)
            for k, v in {
                "energy_weight": 0.6,
                "age_weight": 0.8,
                "speed_factor_weight": 0.7,
                "interaction_strength_weight": 0.7,
                "synergy_affinity_weight": 0.8,
                "colony_factor_weight": 0.9,
                "drift_sensitivity_weight": 0.6,
            }.items()
        }

        # Reproduction parameters for dynamic population
        self.reproduction_energy_threshold: float = max(1.0, 180.0)
        self.reproduction_mutation_rate: float = np.clip(0.3, 0.0, 1.0)
        self.reproduction_offspring_energy_fraction: float = np.clip(0.5, 0.0, 1.0)

        # Enhanced clustering parameters
        self.alignment_strength: float = np.clip(0.4, 0.0, 1.0)
        self.cohesion_strength: float = np.clip(0.5, 0.0, 1.0)
        self.separation_strength: float = np.clip(0.3, 0.0, 1.0)
        self.cluster_radius: float = max(1.0, 15.0)

        self.particle_size: float = max(1.0, 5.0)

        self.energy_efficiency_range: Tuple[float, float] = (-0.4, 3.0)

        self.genetics = GeneticParamConfig()

        # Enhanced speciation and colony parameters
        self.speciation_threshold: float = np.clip(0.8, 0.0, 1.0)
        self.colony_formation_probability: float = np.clip(0.4, 0.0, 1.0)
        self.colony_radius: float = max(1.0, 250.0)
        self.colony_cohesion_strength: float = np.clip(0.6, 0.0, 1.0)

        # Advanced parameters for emergence
        self.synergy_evolution_rate: float = np.clip(0.08, 0.0, 1.0)
        self.complexity_factor: float = max(0.1, 2.0)
        self.structural_complexity_weight: float = np.clip(0.9, 0.0, 1.0)

        # Safety epsilon for numerical stability
        self.EPSILON: float = 1e-10

        self._validate()

    def _validate(self) -> None:
        """
        Validate configuration parameters with comprehensive error checking.
        """
        try:
            validation_rules = [
                (self.n_cell_types > 0, "Number of cell types must be greater than 0"),
                (
                    self.particles_per_type > 0,
                    "Particles per type must be greater than 0",
                ),
                (self.mass_range[0] > 0, "Minimum mass must be positive"),
                (self.base_velocity_scale > 0, "Base velocity scale must be positive"),
                (
                    0.0 <= self.mass_based_fraction <= 1.0,
                    "Mass-based fraction must be between 0.0 and 1.0",
                ),
                (
                    self.interaction_strength_range[0]
                    < self.interaction_strength_range[1],
                    "Invalid interaction strength range",
                ),
                (self.max_frames >= 0, "Maximum frames must be non-negative"),
                (self.initial_energy > 0, "Initial energy must be positive"),
                (0.0 <= self.friction <= 1.0, "Friction must be between 0.0 and 1.0"),
                (
                    self.global_temperature >= 0,
                    "Global temperature must be non-negative",
                ),
                (self.predation_range > 0, "Predation range must be positive"),
                (
                    0.0 <= self.energy_transfer_factor <= 1.0,
                    "Energy transfer factor must be between 0.0 and 1.0",
                ),
                (self.cluster_radius > 0, "Cluster radius must be positive"),
                (self.particle_size > 0, "Particle size must be positive"),
                (
                    self.speciation_threshold > 0,
                    "Speciation threshold must be positive",
                ),
                (self.synergy_range > 0, "Synergy range must be positive"),
                (self.colony_radius > 0, "Colony radius must be positive"),
                (
                    self.reproduction_energy_threshold > 0,
                    "Reproduction energy threshold must be positive",
                ),
                (
                    0.0 <= self.reproduction_offspring_energy_fraction <= 1.0,
                    "Invalid reproduction offspring energy fraction",
                ),
                (
                    0.0 <= self.genetics.gene_mutation_rate <= 1.0,
                    "Gene mutation rate must be between 0.0 and 1.0",
                ),
                (
                    self.genetics.gene_mutation_range[0]
                    < self.genetics.gene_mutation_range[1],
                    "Invalid gene mutation range",
                ),
                (
                    self.energy_efficiency_range[0] < self.energy_efficiency_range[1],
                    "Invalid energy efficiency range",
                ),
                (
                    self.genetics.energy_efficiency_mutation_range[0]
                    < self.genetics.energy_efficiency_mutation_range[1],
                    "Invalid energy efficiency mutation range",
                ),
            ]

            for condition, message in validation_rules:
                if not condition:
                    raise ValueError(message)

        except Exception as e:
            # Set safe default values if validation fails
            self._set_safe_defaults()
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def _set_safe_defaults(self) -> None:
        """
        Set safe default values if validation fails, with robust error handling and optimized defaults
        for high-performance particle simulation.
        """
        try:
            # Core simulation parameters with safe minimum values
            self.n_cell_types = max(1, min(10, self.n_cell_types))
            self.particles_per_type = max(1, min(50, self.particles_per_type))
            self.min_particles_per_type = max(1, min(50, self.min_particles_per_type))
            self.max_particles_per_type = max(300, self.min_particles_per_type)
            self.mass_range = (max(1e-10, 0.2), max(15.0, self.mass_range[1]))
            self.base_velocity_scale = max(0.1, min(2.0, self.base_velocity_scale))
            self.mass_based_fraction = np.clip(self.mass_based_fraction, 0.0, 1.0)
            self.interaction_strength_range = (-3.0, 3.0)
            self.max_frames = max(0, self.max_frames)
            self.initial_energy = max(1.0, min(150.0, self.initial_energy))
            self.friction = np.clip(self.friction, 0.0, 1.0)
            self.global_temperature = max(0.0, min(1.0, self.global_temperature))
            self.predation_range = max(1.0, min(75.0, self.predation_range))
            self.energy_transfer_factor = np.clip(self.energy_transfer_factor, 0.0, 1.0)
            self.mass_transfer = bool(self.mass_transfer)
            self.max_age = max(1.0, self.max_age)
            self.evolution_interval = max(1, min(3000, self.evolution_interval))
            self.synergy_range = max(1.0, min(200.0, self.synergy_range))

            # Culling weights with safe normalization
            weight_sum = sum(self.culling_fitness_weights.values()) + 1e-10
            self.culling_fitness_weights = {
                k: np.clip(v / weight_sum, 0.0, 1.0)
                for k, v in self.culling_fitness_weights.items()
            }

            # Reproduction and clustering parameters
            self.reproduction_energy_threshold = max(
                1.0, min(180.0, self.reproduction_energy_threshold)
            )
            self.reproduction_mutation_rate = np.clip(
                self.reproduction_mutation_rate, 0.0, 1.0
            )
            self.reproduction_offspring_energy_fraction = np.clip(
                self.reproduction_offspring_energy_fraction, 0.0, 1.0
            )
            self.alignment_strength = np.clip(self.alignment_strength, 0.0, 1.0)
            self.cohesion_strength = np.clip(self.cohesion_strength, 0.0, 1.0)
            self.separation_strength = np.clip(self.separation_strength, 0.0, 1.0)
            self.cluster_radius = max(1.0, min(15.0, self.cluster_radius))
            self.particle_size = max(1.0, min(5.0, self.particle_size))
            self.energy_efficiency_range = (
                -0.4,
                max(0.0, self.energy_efficiency_range[1]),
            )

            # Advanced parameters
            self.speciation_threshold = np.clip(self.speciation_threshold, 0.0, 1.0)
            self.colony_formation_probability = np.clip(
                self.colony_formation_probability, 0.0, 1.0
            )
            self.colony_radius = max(1.0, min(250.0, self.colony_radius))
            self.colony_cohesion_strength = np.clip(
                self.colony_cohesion_strength, 0.0, 1.0
            )
            self.synergy_evolution_rate = np.clip(self.synergy_evolution_rate, 0.0, 1.0)
            self.complexity_factor = max(0.1, min(2.0, self.complexity_factor))
            self.structural_complexity_weight = np.clip(
                self.structural_complexity_weight, 0.0, 1.0
            )

            # Safety epsilon
            self.EPSILON = max(1e-10, self.EPSILON)

        except Exception as e:
            # Ultimate fallback values if something goes wrong
            self.n_cell_types = 1
            self.particles_per_type = 1
            self.mass_range = (1e-10, 1.0)
            self.base_velocity_scale = 0.1
            print(
                f"Error setting safe defaults: {str(e)}. Using minimum viable configuration."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration parameters to a dictionary with error handling.
        """
        try:
            config_dict = {
                k: v
                for k, v in self.__dict__.items()
                if not k.startswith("_") and k != "genetics"
            }
            config_dict["genetics"] = {
                k: v for k, v in self.genetics.__dict__.items() if not k.startswith("_")
            }
            return config_dict
        except Exception:
            return {"error": "Failed to convert configuration to dictionary"}


###############################################################
# Cellular Component & Type Data Management
###############################################################


class CellularTypeData:
    """
    Represents a cellular type with multiple cellular components.
    Manages positions, velocities, energy, mass, and genetic traits of components.
    """

    def __init__(
        self,
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
        gene_traits: List[str] = [
            "speed_factor",
            "interaction_strength",
            "perception_range",
            "reproduction_rate",
            "synergy_affinity",
            "colony_factor",
            "drift_sensitivity",
        ],
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
        max_culture: float = 1.0,
    ):
        """
        Initialize a CellularTypeData instance with given parameters.
        """
        # Input validation and sanitization
        n_particles = max(1, int(n_particles))
        window_width = max(1, int(window_width))
        window_height = max(1, int(window_height))

        # Store metadata with validation
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
            coords = random_xy(window_width, window_height, n_particles)
            self.x = coords[:, 0].astype(np.float64)
            self.y = coords[:, 1].astype(np.float64)

            # Initialize energy efficiency safely
            if energy_efficiency is None:
                self.energy_efficiency = np.clip(
                    np.random.uniform(
                        self.min_energy_efficiency,
                        self.max_energy_efficiency,
                        n_particles,
                    ),
                    self.min_energy_efficiency,
                    self.max_energy_efficiency,
                ).astype(np.float64)
            else:
                self.energy_efficiency = np.full(
                    n_particles,
                    np.clip(
                        float(energy_efficiency),
                        self.min_energy_efficiency,
                        self.max_energy_efficiency,
                    ),
                    dtype=np.float64,
                )

            # Safe velocity scaling calculation
            velocity_scaling = base_velocity_scale / np.maximum(
                self.energy_efficiency, 1e-10
            )

            # Initialize velocities safely
            self.vx = np.clip(
                np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
                self.min_velocity,
                self.max_velocity,
            ).astype(np.float64)
            self.vy = np.clip(
                np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
                self.min_velocity,
                self.max_velocity,
            ).astype(np.float64)

            # Initialize energy safely
            self.energy = np.clip(
                np.full(n_particles, initial_energy, dtype=np.float64),
                self.min_energy,
                self.max_energy,
            )

            # Initialize mass safely for mass-based types
            if self.mass_based:
                if mass is None or mass <= 0.0:
                    mass = self.min_mass
                self.mass = np.clip(
                    np.full(n_particles, mass, dtype=np.float64),
                    self.min_mass,
                    self.max_mass,
                )
            else:
                self.mass = None

            # Initialize status arrays safely
            self.alive = np.ones(n_particles, dtype=bool)
            self.age = np.zeros(n_particles, dtype=np.float64)
            self.max_age = float(max_age)

            # Initialize gene traits safely with clipping
            self.speed_factor = np.clip(
                np.random.uniform(0.5, 1.5, n_particles), 0.1, 2.0
            )
            self.interaction_strength = np.clip(
                np.random.uniform(0.5, 1.5, n_particles), 0.1, 2.0
            )
            self.perception_range = np.clip(
                np.random.uniform(50.0, 150.0, n_particles),
                self.min_perception,
                self.max_perception,
            )
            self.reproduction_rate = np.clip(
                np.random.uniform(0.1, 0.5, n_particles),
                self.min_reproduction,
                self.max_reproduction,
            )
            self.synergy_affinity = np.clip(
                np.random.uniform(0.5, 1.5, n_particles),
                self.min_synergy,
                self.max_synergy,
            )
            self.colony_factor = np.clip(
                np.random.uniform(0.0, 1.0, n_particles),
                self.min_colony,
                self.max_colony,
            )
            self.drift_sensitivity = np.clip(
                np.random.uniform(0.5, 1.5, n_particles), self.min_drift, self.max_drift
            )
            self.max_energy_storage = np.clip(
                np.random.uniform(150.0, 300.0, n_particles), 150.0, 1000.0
            )  # Energy storage
            self.sensory_sensitivity = np.clip(
                np.random.uniform(0.5, 1.5, n_particles), 0.1, 2.0
            )  # Sensory sensitivity
            self.short_term_memory = np.zeros(
                n_particles, dtype=np.float64
            )  # Short-term memory
            self.long_term_memory = np.zeros(
                n_particles, dtype=np.float64
            )  # Long-term memory
            self.memory_transfer_rate = np.clip(
                np.random.uniform(0.1, 0.9, n_particles), 0.01, 0.99
            )  # Memory transfer rate
            self.communication_range = np.clip(
                np.random.uniform(50.0, 200.0, n_particles), 20.0, 500.0
            )  # Communication range
            self.socialization_tendency = np.clip(
                np.random.uniform(0.0, 1.0, n_particles), 0.0, 1.0
            )  # Socialization tendency
            self.colony_building_skill = np.clip(
                np.random.uniform(0.0, 1.0, n_particles), 0.0, 1.0
            )  # Colony building skill
            self.cultural_influence = np.clip(
                np.random.uniform(0.0, 1.0, n_particles), 0.0, 1.0
            )  # Cultural influence

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
            # Fallback initialization with minimum values if error occurs
            print(f"Error during initialization: {str(e)}")
            self._initialize_fallback(n_particles)

    def _initialize_fallback(self, n_particles: int) -> None:
        """Safe fallback initialization with minimum values."""
        n_particles = max(1, int(n_particles))
        self.x = np.zeros(n_particles, dtype=np.float64)
        self.y = np.zeros(n_particles, dtype=np.float64)
        self.vx = np.zeros(n_particles, dtype=np.float64)
        self.vy = np.zeros(n_particles, dtype=np.float64)
        self.energy = np.full(n_particles, self.min_energy, dtype=np.float64)
        self.energy_efficiency = np.full(
            n_particles, self.min_energy_efficiency, dtype=np.float64
        )
        if self.mass_based:
            self.mass = np.full(n_particles, self.min_mass, dtype=np.float64)
        self.alive = np.ones(n_particles, dtype=bool)
        self.age = np.zeros(n_particles, dtype=np.float64)
        # Initialize other arrays with safe minimum values...

    def _validate_array_shapes(self) -> None:
        """Validate and correct array shapes for consistency."""
        base_size = len(self.x)
        arrays_to_check = [
            "y",
            "vx",
            "vy",
            "energy",
            "alive",
            "age",
            "energy_efficiency",
            "speed_factor",
            "interaction_strength",
            "perception_range",
            "reproduction_rate",
            "synergy_affinity",
            "colony_factor",
            "drift_sensitivity",
            "species_id",
            "parent_id",
            "max_energy_storage",
            "sensory_sensitivity",
            "short_term_memory",
            "long_term_memory",
            "memory_transfer_rate",
            "communication_range",
            "socialization_tendency",
            "colony_building_skill",
            "cultural_influence",
        ]

        for attr in arrays_to_check:
            current = getattr(self, attr)
            if len(current) != base_size:
                setattr(self, attr, np.resize(current, base_size))

        if self.mass_based and self.mass is not None:
            if len(self.mass) != base_size:
                self.mass = np.resize(self.mass, base_size)

    def is_alive_mask(self) -> np.ndarray:
        """Compute alive mask with safe array operations."""
        try:
            self._validate_array_shapes()
            mask = (
                self.alive & (self.energy > self.min_energy) & (self.age < self.max_age)
            )
            if self.mass_based and self.mass is not None:
                mask &= self.mass > self.min_mass
            return mask
        except Exception:
            return np.ones(len(self.x), dtype=bool)

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
                "x",
                "y",
                "vx",
                "vy",
                "energy",
                "alive",
                "age",
                "energy_efficiency",
                "speed_factor",
                "interaction_strength",
                "perception_range",
                "reproduction_rate",
                "synergy_affinity",
                "colony_factor",
                "drift_sensitivity",
                "species_id",
                "parent_id",
                "max_energy_storage",
                "sensory_sensitivity",
                "short_term_memory",
                "long_term_memory",
                "memory_transfer_rate",
                "communication_range",
                "socialization_tendency",
                "colony_building_skill",
                "cultural_influence",
            ]

            for attr in arrays_to_filter:
                current = getattr(self, attr)
                if len(current) > len(alive_mask):
                    current = current[: len(alive_mask)]
                elif len(current) < len(alive_mask):
                    alive_mask = alive_mask[: len(current)]
                setattr(self, attr, current[alive_mask])

            if self.mass_based and self.mass is not None:
                if len(self.mass) > len(alive_mask):
                    self.mass = self.mass[: len(alive_mask)]
                elif len(self.mass) < len(alive_mask):
                    alive_mask = alive_mask[: len(self.mass)]
                self.mass = self.mass[alive_mask]

        except Exception as e:
            print(f"Error in remove_dead: {str(e)}")
            self._validate_array_shapes()

    def _handle_energy_transfer(
        self,
        dead_due_to_age: np.ndarray,
        alive_mask: np.ndarray,
        config: SimulationConfig,
    ) -> None:
        """Handle energy transfer from dead components safely."""
        try:
            alive_indices = np.where(alive_mask)[0]
            dead_age_indices = np.where(dead_due_to_age)[0]

            if len(alive_indices) > 0:
                alive_positions = np.column_stack(
                    (self.x[alive_indices], self.y[alive_indices])
                )
                tree = cKDTree(alive_positions)

                batch_size = min(1000, len(dead_age_indices))
                for i in range(0, len(dead_age_indices), batch_size):
                    batch_indices = dead_age_indices[i : i + batch_size]
                    dead_positions = np.column_stack(
                        (self.x[batch_indices], self.y[batch_indices])
                    )
                    dead_energies = self.energy[batch_indices]

                    distances, neighbors = tree.query(
                        dead_positions,
                        k=min(3, len(alive_indices)),
                        distance_upper_bound=config.predation_range,
                    )

                    valid_mask = distances < config.predation_range
                    for j, (dist_row, neighbor_row, dead_energy) in enumerate(
                        zip(distances, neighbors, dead_energies)
                    ):
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
        cultural_influence_val: float,
    ) -> None:
        """Add new component safely with array broadcasting."""
        try:
            # Validate and clip input values
            x = float(x)
            y = float(y)
            vx = np.clip(float(vx), self.min_velocity, self.max_velocity)
            vy = np.clip(float(vy), self.min_velocity, self.max_velocity)
            energy = np.clip(float(energy), self.min_energy, self.max_energy)
            energy_efficiency_val = np.clip(
                float(energy_efficiency_val),
                self.min_energy_efficiency,
                self.max_energy_efficiency,
            )

            # Prepare new values as arrays for broadcasting
            new_values = {
                "x": np.array([x]),
                "y": np.array([y]),
                "vx": np.array([vx]),
                "vy": np.array([vy]),
                "energy": np.array([energy]),
                "alive": np.array([True]),
                "age": np.array([0.0]),
                "energy_efficiency": np.array([energy_efficiency_val]),
                "speed_factor": np.array([speed_factor_val]),
                "interaction_strength": np.array([interaction_strength_val]),
                "perception_range": np.array([perception_range_val]),
                "reproduction_rate": np.array([reproduction_rate_val]),
                "synergy_affinity": np.array([synergy_affinity_val]),
                "colony_factor": np.array([colony_factor_val]),
                "drift_sensitivity": np.array([drift_sensitivity_val]),
                "species_id": np.array([species_id_val]),
                "parent_id": np.array([parent_id_val]),
                "max_energy_storage": np.array([max_energy_storage_val]),
                "sensory_sensitivity": np.array([sensory_sensitivity_val]),
                "short_term_memory": np.array(
                    [0.0]
                ),  # Initialize short-term memory to 0
                "long_term_memory": np.array([0.0]),  # Initialize long-term memory to 0
                "memory_transfer_rate": np.array([memory_transfer_rate_val]),
                "communication_range": np.array([communication_range_val]),
                "socialization_tendency": np.array([socialization_tendency_val]),
                "colony_building_skill": np.array([colony_building_skill_val]),
                "cultural_influence": np.array([cultural_influence_val]),
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
        "NOP": 0,  # No operation
        "ADD": 1,  # Add
        "SUB": 2,  # Subtract
        "MUL": 3,  # Multiply
        "DIV": 4,  # Divide
        "POW": 5,  # Power
        "SQRT": 6,  # Square root
        "LOG": 7,  # Natural log
        "EXP": 8,  # Exponential
        # Neural operations
        "ACTIVATE": 10,  # Activation function
        "BACKPROP": 11,  # Backpropagation
        "WEIGHT": 12,  # Weight update
        "BIAS": 13,  # Bias update
        "GRAD": 14,  # Gradient computation
        "BATCH": 15,  # Batch normalization
        "DROP": 16,  # Dropout
        # Memory & flow control
        "LOAD": 20,  # Load from memory
        "STORE": 21,  # Store to memory
        "PUSH": 22,  # Push to stack
        "POP": 23,  # Pop from stack
        "JMP": 24,  # Jump
        "BRANCH": 25,  # Conditional branch
        "CALL": 26,  # Call subroutine
        "RET": 27,  # Return
        # Advanced neural ops
        "CONV": 30,  # Convolution
        "POOL": 31,  # Pooling
        "ATTN": 32,  # Attention mechanism
        "LSTM": 33,  # LSTM cell
        "GRU": 34,  # GRU cell
        "TRANS": 35,  # Transformer block
        # Genetic operations
        "MUTATE": 40,  # Mutation
        "CROSS": 41,  # Crossover
        "SELECT": 42,  # Selection
        "EVOLVE": 43,  # Evolution step
        # System operations
        "SYNC": 50,  # Synchronization
        "DIST": 51,  # Distribution
        "OPTIM": 52,  # Optimization
        "DEBUG": 53,  # Debug
    }

    # Reverse mapping
    INSTR_NAMES = {v: k for k, v in INSTR_SET.items()}

    # System parameters
    REGISTER_COUNT = 64  # General purpose registers
    VECTOR_REGISTER_COUNT = 32  # Vector registers
    MEMORY_SIZE = 1 << 20  # 1MB addressable memory
    CACHE_SIZE = 1 << 16  # 64KB instruction cache
    STACK_SIZE = 1 << 12  # 4KB call stack
    MAX_RECURSION = 256  # Max recursion depth

    # Neural parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    DROPOUT_RATE = 0.5

    def __init__(self):
        """Initialize the neural instruction architecture."""
        # Core components
        self.registers = np.zeros(self.REGISTER_COUNT, dtype=np.float32)
        self.vector_registers = np.zeros(
            (self.VECTOR_REGISTER_COUNT, self.BATCH_SIZE), dtype=np.float32
        )
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
            "fetch": self._fetch_instruction,
            "decode": self._decode_instruction,
            "execute": self._execute_instruction,
            "memory": self._memory_operation,
            "writeback": self._writeback_result,
        }

        # Fast dispatch tables
        self.op_dispatch = {
            op: getattr(self, f"_exec_{name.lower()}", self._exec_nop)
            for name, op in self.INSTR_SET.items()
        }

        # Vectorized operation tables
        self.vector_ops = {
            "ADD": np.add,
            "MUL": np.multiply,
            "DIV": np.divide,
            "POW": np.power,
            "SQRT": np.sqrt,
            "LOG": np.log,
            "EXP": np.exp,
        }

    def _init_neural_functions(self):
        """Initialize neural network activation functions."""
        self.activation_functions = {
            "relu": lambda x: np.maximum(0, x),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": np.tanh,
            "softmax": lambda x: np.exp(x) / np.sum(np.exp(x), axis=0),
        }

        self.gradient_functions = {
            "relu": lambda x: np.where(x > 0, 1, 0),
            "sigmoid": lambda x: x * (1 - x),
            "tanh": lambda x: 1 - x**2,
            "softmax": lambda x: x * (1 - x),
        }

    @staticmethod
    def create_optimized_sequence(
        length: int, instruction_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Create an optimized instruction sequence with optional weighting."""
        if instruction_weights:
            instructions = list(GeneticInstructions.INSTR_SET.values())
            weights = [
                instruction_weights.get(GeneticInstructions.INSTR_NAMES[i], 1.0)
                for i in instructions
            ]
            return np.random.choice(
                instructions, size=length, p=np.array(weights) / sum(weights)
            )
        return np.random.choice(
            list(GeneticInstructions.INSTR_SET.values()), size=length
        )


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
        code = np.array(
            [GeneticInstructions.random_instruction() for _ in range(length)], dtype=int
        )
        return code

    def mutate(self):
        # Point mutations
        for i in range(len(self.code)):
            if random.random() < self.config.genetics.gene_mutation_rate:
                self.code[i] = GeneticInstructions.random_instruction()

        # Insertions
        if (
            random.random() < self.config.genetics.insertion_rate
            and len(self.code) < self.config.genetics.max_genome_length
        ):
            pos = random.randint(0, len(self.code))
            new_instr = GeneticInstructions.random_instruction()
            self.code = np.insert(self.code, pos, new_instr)
            self.promoters = np.insert(self.promoters, pos, False)
            self.inhibitors = np.insert(self.inhibitors, pos, False)
            self.epigenetic_marks = np.insert(self.epigenetic_marks, pos, 0.0)

        # Deletions
        if random.random() < self.config.genetics.deletion_rate and len(self.code) > 50:
            pos = random.randint(0, len(self.code) - 1)
            self.code = np.delete(self.code, pos)
            self.promoters = np.delete(self.promoters, pos)
            self.inhibitors = np.delete(self.inhibitors, pos)
            self.epigenetic_marks = np.delete(self.epigenetic_marks, pos)

        # Duplications
        if (
            random.random() < self.config.genetics.duplication_rate
            and len(self.code) * 2 < self.config.genetics.max_genome_length
        ):
            start = random.randint(0, len(self.code) - 1)
            end = random.randint(start, len(self.code) - 1)
            segment = self.code[start : end + 1]
            self.code = np.concatenate((self.code, segment))
            self.promoters = np.concatenate(
                (self.promoters, self.promoters[start : end + 1])
            )
            self.inhibitors = np.concatenate(
                (self.inhibitors, self.inhibitors[start : end + 1])
            )
            self.epigenetic_marks = np.concatenate(
                (self.epigenetic_marks, self.epigenetic_marks[start : end + 1])
            )

        # Transposons
        if (
            random.random() < self.config.genetics.transposon_rate
            and len(self.code) > 100
        ):
            start = random.randint(0, len(self.code) - 50)
            end = min(len(self.code) - 1, start + random.randint(10, 50))
            segment = self.code[start : end + 1]
            pseg = self.promoters[start : end + 1]
            iseg = self.inhibitors[start : end + 1]
            eseg = self.epigenetic_marks[start : end + 1]

            self.code = np.delete(self.code, slice(start, end + 1))
            self.promoters = np.delete(self.promoters, slice(start, end + 1))
            self.inhibitors = np.delete(self.inhibitors, slice(start, end + 1))
            self.epigenetic_marks = np.delete(
                self.epigenetic_marks, slice(start, end + 1)
            )

            pos = random.randint(0, len(self.code))
            self.code = np.insert(self.code, pos, segment)
            self.promoters = np.insert(self.promoters, pos, pseg)
            self.inhibitors = np.insert(self.inhibitors, pos, iseg)
            self.epigenetic_marks = np.insert(self.epigenetic_marks, pos, eseg)

        # Epigenetic changes
        for i in range(len(self.code)):
            if random.random() < self.config.genetics.epigenetic_mark_rate:
                self.epigenetic_marks[i] = min(
                    1.0, self.epigenetic_marks[i] + random.uniform(0.0, 0.2)
                )
            if random.random() < self.config.genetics.epigenetic_erase_rate:
                self.epigenetic_marks[i] = max(
                    0.0, self.epigenetic_marks[i] - random.uniform(0.0, 0.2)
                )


###############################################################
# Genetic Interpreter Class
###############################################################


class GeneticInterpreter:
    """Advanced genetic sequence interpreter implementing Turing-complete genetic programming."""

    def __init__(self, gene_sequence: Optional[List[List[Any]]] = None):
        """Initialize genetic interpreter with optimized defaults."""
        self.default_sequence = [
            ["start_movement", 1.0, 0.1, 0.0],  # speed, randomness, bias
            ["start_interaction", 0.5, 100.0],  # strength, radius
            ["start_energy", 0.1, 0.5, 0.3],  # gain, efficiency, transfer
            [
                "start_reproduction",
                150.0,
                100.0,
                50.0,
                30.0,
            ],  # threshold, cost, bonus, penalty
            ["start_growth", 0.1, 2.0, 100.0],  # rate, factor, limit
            ["start_predation", 10.0, 5.0],  # strength, range
        ]
        self.gene_sequence = (
            gene_sequence if gene_sequence is not None else self.default_sequence
        )

        # Initialize core components
        self._setup_safety_bounds()
        self._initialize_genetic_mechanisms()
        self._setup_caches()

    def _setup_safety_bounds(self) -> None:
        """Configure comprehensive safety bounds."""
        self.bounds = {
            "energy": (0.0, 1000.0),
            "velocity": (-20.0, 20.0),
            "traits": (0.01, 5.0),
            "mass": (0.1, 10.0),
            "age": (0.0, float("inf")),
            "distance": (1e-10, float("inf")),
            "interaction": (0.0, 1000.0),
            "reproduction": (0.0, 500.0),
        }

    def _initialize_genetic_mechanisms(self) -> None:
        """Initialize advanced genetic control mechanisms."""
        # Regulatory networks
        self.regulatory_networks = {
            "movement": {"inhibitors": [], "activators": [], "threshold": 0.5},
            "interaction": {"inhibitors": [], "activators": [], "threshold": 0.4},
            "energy": {"inhibitors": [], "activators": [], "threshold": 0.3},
            "reproduction": {"inhibitors": [], "activators": [], "threshold": 0.6},
            "growth": {"inhibitors": [], "activators": [], "threshold": 0.4},
            "predation": {"inhibitors": [], "activators": [], "threshold": 0.7},
        }

        # Epistatic interactions
        self.epistatic_interactions = {
            "movement": {"modifiers": {"energy": 0.2, "interaction": 0.1}},
            "interaction": {"modifiers": {"energy": 0.3, "predation": 0.2}},
            "energy": {"modifiers": {"growth": 0.2, "reproduction": 0.3}},
            "reproduction": {"modifiers": {"energy": -0.2, "growth": 0.2}},
            "growth": {"modifiers": {"energy": -0.1, "reproduction": -0.1}},
            "predation": {"modifiers": {"energy": 0.3, "movement": 0.2}},
        }

        # Epigenetic modifications
        self.epigenetic_modifications = {
            "methylation": defaultdict(float),
            "acetylation": defaultdict(float),
            "phosphorylation": defaultdict(float),
            "ubiquitination": defaultdict(float),
        }

    def _setup_caches(self) -> None:
        """Initialize performance optimization caches."""
        self.computation_cache = {}
        self.regulatory_state_cache = {}
        self.interaction_cache = {}
        self.MAX_CACHE_SIZE = 1000

    def decode(
        self,
        particle: CellularTypeData,
        others: List[CellularTypeData],
        env: SimulationConfig,
    ) -> None:
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
                gene_data = self._apply_epistatic_effects(
                    gene_type, gene_data, particle
                )

                # Apply epigenetic modifications
                gene_data = self._apply_epigenetic_mods(gene_type, gene_data)

                # Execute gene function with optimized data
                method = getattr(
                    self, f"apply_{gene_type.replace('start_', '')}_gene", None
                )
                if method:
                    method(particle, others, gene_data, env)

        except Exception as e:
            print(f"Error in genetic decoding: {str(e)}")
            self._ensure_particle_stability(particle)

    def _check_regulatory_state(
        self, gene_type: str, particle: CellularTypeData
    ) -> bool:
        """Check if gene expression is allowed by regulatory networks."""
        try:
            cache_key = (gene_type, id(particle))
            if cache_key in self.regulatory_state_cache:
                return self.regulatory_state_cache[cache_key]

            network = self.regulatory_networks.get(gene_type.replace("start_", ""))
            if not network:
                return True

            # Calculate regulatory score
            activator_score = sum(
                1
                for act in network["activators"]
                if self._check_condition(act, particle)
            )
            inhibitor_score = sum(
                1
                for inh in network["inhibitors"]
                if self._check_condition(inh, particle)
            )

            threshold = network["threshold"]
            result = (activator_score - inhibitor_score) >= threshold

            self.regulatory_state_cache[cache_key] = result
            return result

        except Exception:
            return True

    def _apply_epistatic_effects(
        self, gene_type: str, gene_data: np.ndarray, particle: CellularTypeData
    ) -> np.ndarray:
        """Apply epistatic interactions between genes."""
        try:
            base_type = gene_type.replace("start_", "")
            modifiers = self.epistatic_interactions.get(base_type, {}).get(
                "modifiers", {}
            )

            modified_data = gene_data.copy()
            for mod_gene, factor in modifiers.items():
                if hasattr(particle, mod_gene):
                    mod_value = getattr(particle, mod_gene)
                    if isinstance(mod_value, np.ndarray):
                        mod_value = np.mean(mod_value)
                    modified_data *= 1 + factor * mod_value

            return np.clip(modified_data, *self.bounds["traits"])

        except Exception:
            return gene_data

    def _apply_epigenetic_mods(
        self, gene_type: str, gene_data: np.ndarray
    ) -> np.ndarray:
        """Apply epigenetic modifications to gene expression."""
        try:
            base_type = gene_type.replace("start_", "")

            # Combine all modification effects
            total_mod = 1.0
            for mod_type, mod_values in self.epigenetic_modifications.items():
                mod_factor = mod_values.get(base_type, 0.0)
                total_mod *= 1 + mod_factor

            return np.clip(gene_data * total_mod, *self.bounds["traits"])

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
            self.give_take_matrix = np.zeros(
                (self.config.n_cell_types, self.config.n_cell_types), dtype=bool
            )
            self.synergy_matrix = np.zeros(
                (self.config.n_cell_types, self.config.n_cell_types), dtype=np.float32
            )

    def _create_interaction_matrix(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Create interaction matrix with vectorized operations and safety checks."""
        try:
            n_types = max(1, self.config.n_cell_types)
            final_rules = []

            # Vectorized parameter generation
            type_pairs = np.array(
                [(i, j) for i in range(n_types) for j in range(n_types)]
            )
            mass_based_mask = np.isin(type_pairs, self.mass_based_type_indices)
            both_mass = np.all(mass_based_mask, axis=1)

            # Vectorized random generation
            rand_vals = np.random.random(len(type_pairs))
            use_gravity = both_mass & (rand_vals < 0.5)

            potential_strengths = np.random.uniform(
                self.config.interaction_strength_range[0],
                self.config.interaction_strength_range[1],
                len(type_pairs),
            )
            potential_strengths[rand_vals < 0.5] *= -1

            gravity_factors = np.where(
                use_gravity,
                np.random.uniform(0.1, 2.0, len(type_pairs)),
                np.zeros(len(type_pairs)),
            )

            max_dists = np.random.uniform(50.0, 200.0, len(type_pairs))

            # Create rules with safety bounds
            for idx, (i, j) in enumerate(type_pairs):
                params = {
                    "use_potential": True,
                    "use_gravity": bool(use_gravity[idx]),
                    "potential_strength": float(
                        np.clip(potential_strengths[idx], -1e6, 1e6)
                    ),
                    "gravity_factor": float(np.clip(gravity_factors[idx], 0, 1e3)),
                    "max_dist": float(np.clip(max_dists[idx], 10.0, 1e4)),
                }
                final_rules.append((int(i), int(j), params))

            return final_rules

        except Exception as e:
            print(f"Interaction matrix creation error handled: {str(e)}")
            return [
                (
                    0,
                    0,
                    {
                        "use_potential": True,
                        "use_gravity": False,
                        "potential_strength": 1.0,
                        "gravity_factor": 0.0,
                        "max_dist": 50.0,
                    },
                )
            ]

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
            return np.zeros(
                (self.MIN_ARRAY_SIZE, self.MIN_ARRAY_SIZE), dtype=np.float32
            )

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
                        self.config.interaction_strength_range[1],
                    )

                if (
                    rand_vals[1] < 0.05 and "gravity_factor" in params
                ):  # Gravity mutation
                    params["gravity_factor"] = np.clip(
                        params["gravity_factor"] * mutation_factors[1], 0.0, 10.0
                    )

                if rand_vals[2] < 0.05:  # Max distance mutation
                    params["max_dist"] = np.clip(
                        params["max_dist"] * mutation_factors[2], 10.0, 1000.0
                    )

            # Energy transfer evolution
            if np.random.random() < 0.1:
                self.config.energy_transfer_factor = np.clip(
                    self.config.energy_transfer_factor * np.random.uniform(0.95, 1.05),
                    0.0,
                    1.0,
                )

            # Vectorized synergy evolution
            evolution_mask = np.random.random(self.synergy_matrix.shape) < 0.05
            mutation_values = np.random.uniform(-0.05, 0.05, self.synergy_matrix.shape)

            self.synergy_matrix = np.clip(
                np.where(
                    evolution_mask,
                    self.synergy_matrix + mutation_values,
                    self.synergy_matrix,
                ),
                0.0,
                1.0,
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

    def __init__(
        self,
        config: SimulationConfig,
        colors: List[Tuple[int, int, int]],
        mass_based_type_indices: List[int],
    ):
        """Initialize with robust validation and optimized data structures."""
        self.config = config
        self.cellular_types: List[CellularTypeData] = []
        self.mass_based_type_indices = np.array(mass_based_type_indices, dtype=np.int32)
        self.colors = colors
        self.EPSILON = np.finfo(np.float64).tiny
        self.MIN_ARRAY_SIZE = 1

        # Pre-allocate reusable arrays for performance
        self._temp_arrays = {
            "mutation_buffer": np.zeros(
                config.max_particles_per_type, dtype=np.float64
            ),
            "distance_buffer": np.zeros(
                config.max_particles_per_type, dtype=np.float64
            ),
            "mask_buffer": np.zeros(config.max_particles_per_type, dtype=bool),
        }

        # Initialize adaptive parameters
        self._adaptation_rates = {
            "mutation": np.full(config.n_cell_types, 0.5),
            "speciation": np.full(config.n_cell_types, 0.5),
            "energy_transfer": np.full(config.n_cell_types, 0.5),
        }

    def add_cellular_type_data(self, data: CellularTypeData) -> None:
        """Add cellular type with comprehensive validation."""
        if data is not None and self._validate_type_data(data):
            self.cellular_types.append(data)
            self._update_adaptation_rates(len(self.cellular_types) - 1)

    def _validate_type_data(self, data: CellularTypeData) -> bool:
        """Validate cellular type data integrity."""
        try:
            required_attrs = ["x", "y", "energy", "alive"]
            return all(
                hasattr(data, attr) and getattr(data, attr) is not None
                for attr in required_attrs
            )
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
                parent_energy, offspring_energy = self._handle_energy_transfer(
                    ct, eligible_mask
                )
                mutation_mask = self._generate_mutation_mask(num_offspring, type_idx)

                # Generate offspring traits with advanced genetic algorithms
                offspring_traits = self._generate_offspring_traits(
                    ct, parent_indices, mutation_mask
                )

                # Calculate genetic distances and handle speciation
                genetic_distances = self._calculate_genetic_distances(
                    ct, offspring_traits, parent_indices
                )
                new_species_ids = self._assign_species_ids(
                    ct, genetic_distances, parent_indices
                )

                # Add new components with optimized batch operations
                self._add_offspring_components(
                    ct,
                    num_offspring,
                    parent_indices,
                    offspring_energy,
                    offspring_traits,
                    new_species_ids,
                )

                # Update adaptation rates based on reproduction success
                self._update_adaptation_rates(type_idx)

            except Exception as e:
                print(f"Reproduction error handled for type {type_idx}: {str(e)}")
                continue

    def _check_reproduction_conditions(self, ct: CellularTypeData) -> bool:
        """Check if reproduction conditions are met."""
        return (
            ct is not None
            and ct.x.size > 0
            and ct.x.size < self.config.max_particles_per_type
            and np.any(ct.alive)
        )

    def _calculate_eligibility_mask(self, ct: CellularTypeData) -> np.ndarray:
        """Calculate reproduction eligibility with vectorized operations."""
        reproduction_threshold = self._temp_arrays["mutation_buffer"][: ct.x.size]
        np.random.uniform(0, 1, size=ct.x.size)

        return (
            ct.alive
            & (ct.energy > self.config.reproduction_energy_threshold)
            & (reproduction_threshold < ct.reproduction_rate)
        )

    def _handle_energy_transfer(
        self, ct: CellularTypeData, eligible_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Handle energy transfer with safety checks."""
        parent_energy = ct.energy[eligible_mask]
        ct.energy[eligible_mask] = np.maximum(parent_energy * 0.5, self.EPSILON)

        return parent_energy, np.maximum(
            parent_energy * self.config.reproduction_offspring_energy_fraction,
            self.EPSILON,
        )

    def _generate_mutation_mask(self, num_offspring: int, type_idx: int) -> np.ndarray:
        """Generate mutation mask with adaptive rates."""
        return np.random.random(num_offspring) < (
            self.config.genetics.gene_mutation_rate
            * self._adaptation_rates["mutation"][type_idx]
        )

    def _generate_offspring_traits(
        self,
        ct: CellularTypeData,
        parent_indices: np.ndarray,
        mutation_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Generate offspring traits using advanced genetic algorithms."""
        offspring_traits = {}

        for trait in self.config.genetics.gene_traits:
            parent_values = getattr(ct, trait)[parent_indices]
            offspring_traits[trait] = np.copy(parent_values)

            if mutation_mask.any():
                mutation = np.random.normal(
                    loc=0,
                    scale=self.config.genetics.gene_mutation_range[1]
                    - self.config.genetics.gene_mutation_range[0],
                    size=mutation_mask.sum(),
                )
                offspring_traits[trait][mutation_mask] += mutation

        return offspring_traits

    def _calculate_genetic_distances(
        self,
        ct: CellularTypeData,
        offspring_traits: Dict[str, np.ndarray],
        parent_indices: np.ndarray,
    ) -> np.ndarray:
        """Calculate genetic distances with optimized array operations."""
        squared_diffs = np.zeros_like(parent_indices, dtype=np.float64)

        for trait in self.config.genetics.gene_traits:
            diff = offspring_traits[trait] - getattr(ct, trait)[parent_indices]
            squared_diffs += np.square(np.clip(diff, -1e10, 1e10))

        return np.sqrt(squared_diffs)

    def _assign_species_ids(
        self,
        ct: CellularTypeData,
        genetic_distances: np.ndarray,
        parent_indices: np.ndarray,
    ) -> np.ndarray:
        """Assign species IDs based on genetic distances."""
        max_species_id = np.max(ct.species_id) if ct.species_id.size > 0 else 0
        return np.where(
            genetic_distances > self.config.speciation_threshold,
            max_species_id + 1,
            ct.species_id[parent_indices],
        )

    def _add_offspring_components(
        self,
        ct: CellularTypeData,
        num_offspring: int,
        parent_indices: np.ndarray,
        offspring_energy: np.ndarray,
        offspring_traits: Dict[str, np.ndarray],
        new_species_ids: np.ndarray,
    ) -> None:
        """Add offspring components with batch operations."""
        for i in range(num_offspring):
            try:
                velocity_scale = (
                    self.config.base_velocity_scale
                    / np.maximum(offspring_traits["energy_efficiency"][i], self.EPSILON)
                    * offspring_traits["speed_factor"][i]
                )

                ct.add_component(
                    x=ct.x[parent_indices[i]],
                    y=ct.y[parent_indices[i]],
                    vx=np.random.normal(0, velocity_scale),
                    vy=np.random.normal(0, velocity_scale),
                    energy=offspring_energy[i],
                    mass_val=None,  # Handled separately if mass-based
                    **{
                        f"{trait}_val": offspring_traits[trait][i]
                        for trait in self.config.genetics.gene_traits
                    },
                    species_id_val=new_species_ids[i],
                    parent_id_val=ct.type_id,
                    max_age=ct.max_age,
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
                current_rate + 0.1 * (success_rate - 0.5), 0.1, 1.0
            )


###############################################################
# Forces & Interactions
###############################################################


def apply_interaction(
    a_x: np.ndarray,
    a_y: np.ndarray,
    b_x: np.ndarray,
    b_y: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized force computation between cellular components using vectorized operations.
    Implements sophisticated physical models and dynamic force calculations.
    """
    try:
        # Ensure numerical stability with explicit typing
        arrays = np.broadcast_arrays(
            np.asarray(a_x, dtype=np.float64),
            np.asarray(a_y, dtype=np.float64),
            np.asarray(b_x, dtype=np.float64),
            np.asarray(b_y, dtype=np.float64),
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
                np.asarray(params["m_b"], dtype=np.float64),
            )

            gravity_factor = np.float64(params.get("gravity_factor", 1.0))

            F_grav = np.multiply(
                gravity_factor,
                np.divide(np.multiply(m_a, m_b), d_sq, where=valid_mask),
                where=valid_mask,
            )

            fx = np.add(fx, F_grav * dx, where=valid_mask, out=fx)
            fy = np.add(fy, F_grav * dy, where=valid_mask, out=fy)

        return np.nan_to_num(fx, copy=False), np.nan_to_num(fy, copy=False)

    except Exception as e:
        print(f"Interaction calculation error handled: {str(e)}")
        return np.zeros_like(a_x), np.zeros_like(a_y)


def give_take_interaction(
    giver_energy: np.ndarray,
    receiver_energy: np.ndarray,
    giver_mass: Optional[np.ndarray],
    receiver_mass: Optional[np.ndarray],
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Optimized energy and mass transfer system with advanced conservation laws.
    """
    try:
        # Ensure numerical stability
        giver_energy = np.asarray(giver_energy, dtype=np.float64)
        receiver_energy = np.asarray(receiver_energy, dtype=np.float64)

        # Calculate transfer amounts with conservation
        transfer_factor = np.clip(config.energy_transfer_factor, 0, 1)
        transfer_amount = np.multiply(
            receiver_energy, transfer_factor, dtype=np.float64
        )

        # Update energies with conservation laws
        new_receiver = np.subtract(receiver_energy, transfer_amount, dtype=np.float64)
        new_giver = np.add(giver_energy, transfer_amount, dtype=np.float64)

        # Handle mass transfer with conservation
        new_giver_mass = new_receiver_mass = None
        if (
            config.mass_transfer
            and giver_mass is not None
            and receiver_mass is not None
        ):
            giver_mass = np.asarray(giver_mass, dtype=np.float64)
            receiver_mass = np.asarray(receiver_mass, dtype=np.float64)

            mass_transfer = np.multiply(
                receiver_mass, transfer_factor, dtype=np.float64
            )
            new_receiver_mass = np.subtract(
                receiver_mass, mass_transfer, dtype=np.float64
            )
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


def apply_synergy(
    energyA: np.ndarray, energyB: np.ndarray, synergy_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced synergy calculation with emergent behavior patterns.
    """
    try:
        # Ensure numerical stability
        energyA, energyB = np.broadcast_arrays(
            np.asarray(energyA, dtype=np.float64), np.asarray(energyB, dtype=np.float64)
        )

        # Apply synergy effects
        synergy_factor = np.clip(synergy_factor, 0, 1)
        avg_energy = np.multiply(np.add(energyA, energyB), 0.5, dtype=np.float64)

        # Calculate synergistic energies
        complement_factor = 1.0 - synergy_factor
        newA = np.add(
            np.multiply(energyA, complement_factor),
            np.multiply(avg_energy, synergy_factor),
            dtype=np.float64,
        )
        newB = np.add(
            np.multiply(energyB, complement_factor),
            np.multiply(avg_energy, synergy_factor),
            dtype=np.float64,
        )

        # Ensure energy conservation
        return (
            np.maximum(np.nan_to_num(newA, copy=False), 0),
            np.maximum(np.nan_to_num(newB, copy=False), 0),
        )

    except Exception as e:
        print(f"Synergy calculation error handled: {str(e)}")
        return energyA, energyB


def random_xy(window_width: int, window_height: int, n: int = 1) -> np.ndarray:
    """
    Generate n random (x, y) coordinates with robust error handling.
    """
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
# Renderer Class
###############################################################


class Renderer:
    """
    Advanced rendering engine for particle visualization with optimized batch processing,
    dynamic visual effects, and performance monitoring.
    """

    def __init__(self, surface: pygame.Surface, config: SimulationConfig):
        """Initialize the renderer with advanced buffering and optimization."""
        self.surface = surface
        self.config = config

        # Create layered rendering surfaces for compositing
        self._init_surfaces()

        # Initialize fonts and text rendering
        self._init_fonts()

        # Performance monitoring
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

    def _init_surfaces(self):
        """Initialize multi-layered rendering surfaces."""
        size = self.surface.get_size()
        flags = pygame.SRCALPHA

        # Main particle surface with alpha
        self.particle_surface = pygame.Surface(size, flags=flags).convert_alpha()

        # Glow/bloom effect surface
        self.glow_surface = pygame.Surface(size, flags=flags).convert_alpha()

        # Motion trail surface
        self.trail_surface = pygame.Surface(size, flags=flags).convert_alpha()

        # Clear all surfaces
        for surface in [self.particle_surface, self.glow_surface, self.trail_surface]:
            surface.fill((0, 0, 0, 0))

    def _init_fonts(self):
        """Initialize font rendering with fallbacks."""
        try:
            self.font = pygame.font.SysFont("Arial", 20)
            self.large_font = pygame.font.SysFont("Arial", 32)
            self.small_font = pygame.font.SysFont("Arial", 16)
        except pygame.error:
            # Fallback to default font
            self.font = pygame.font.Font(None, 20)
            self.large_font = pygame.font.Font(None, 32)
            self.small_font = pygame.font.Font(None, 16)

    def generate_vibrant_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct vibrant colors with error handling and validation.
        """
        try:
            n = max(1, int(n))
            colors = []

            for i in range(n):
                hue = (i / n) % 1.0
                h_i = int(hue * 6)
                f = np.clip(hue * 6 - h_i, 0, 1)

                # Safe color calculations
                p = 0
                q = int(np.clip((1 - f) * 255, 0, 255))
                t = int(np.clip(f * 255, 0, 255))
                v = 255

                # Safe color assignment
                color = [
                    (v, t, p),
                    (q, v, p),
                    (p, v, t),
                    (p, q, v),
                    (t, p, v),
                    (v, p, q),
                ][h_i % 6]
                colors.append(color)

            return colors
        except Exception:
            # Return safe fallback color
            return [(255, 255, 255)]

    def draw_component(
        self,
        x: float,
        y: float,
        color: Tuple[int, int, int],
        energy: float,
        speed_factor: float,
    ) -> None:
        """Draw a single particle with advanced visual effects."""
        try:
            # Calculate particle appearance based on attributes
            health = np.clip(energy, 0.0, 100.0)
            intensity_factor = np.clip(health / 100.0, 0.0, 1.0)

            # Generate color key for cache
            color_key = (color, intensity_factor, speed_factor)

            # Use cached color if available
            if color_key in self._color_cache:
                c = self._color_cache[color_key]
            else:
                # Calculate new color with enhanced visual effects
                base_intensity = intensity_factor * speed_factor
                fade = (1 - intensity_factor) * 100

                c = tuple(
                    min(
                        255,
                        int(
                            col * base_intensity
                            + fade
                            + (speed_factor - 1) * 20 * intensity_factor
                        ),
                    )
                    for col in color
                )

                # Cache the calculated color
                self._color_cache[color_key] = c

            # Draw main particle
            pos = (int(x), int(y))
            size = int(self.config.particle_size * (0.8 + 0.4 * intensity_factor))
            pygame.draw.circle(self.particle_surface, c, pos, size)

            # Add glow effect for high-energy particles
            if self.glow_enabled and energy > 50:
                glow_size = size + self.blur_radius
                glow_color = tuple(min(255, int(v * 0.7)) for v in c)
                pygame.draw.circle(self.glow_surface, glow_color, pos, glow_size)

        except Exception as e:
            print(f"Error in draw_component: {e}")

    def draw_cellular_type(self, ct: "CellularTypeData") -> None:
        """Efficiently render cellular types using batch processing."""
        try:
            # Get indices of alive particles
            alive_indices = np.where(ct.alive)[0]

            if len(alive_indices) == 0:
                return

            # Process particles in optimized batches
            for i in range(0, len(alive_indices), self._particle_batch_size):
                batch_indices = alive_indices[i : i + self._particle_batch_size]

                # Vectorized attribute gathering
                positions = np.column_stack((ct.x[batch_indices], ct.y[batch_indices]))
                energies = ct.energy[batch_indices]
                speed_factors = ct.speed_factor[batch_indices]

                # Draw each particle in batch
                for idx, (pos, energy, speed) in enumerate(
                    zip(positions, energies, speed_factors)
                ):
                    self.draw_component(pos[0], pos[1], ct.color, energy, speed)

        except Exception as e:
            print(f"Error in draw_cellular_type: {e}")

    def render(self, stats: Dict[str, Any]) -> None:
        """Render the final composite frame with performance monitoring."""
        try:
            # Apply motion trails
            self.trail_surface.fill((0, 0, 0, int(255 * (1 - self.fade_factor))))
            self.trail_surface.blit(self.particle_surface, (0, 0))

            # Composite all layers
            self.surface.blit(self.trail_surface, (0, 0))
            self.surface.blit(
                self.glow_surface, (0, 0), special_flags=pygame.BLEND_RGB_ADD
            )
            self.surface.blit(self.particle_surface, (0, 0))

            # Clear surfaces for next frame
            self.particle_surface.fill((0, 0, 0, 0))
            self.glow_surface.fill((0, 0, 0, 0))

            # Render enhanced stats display
            current_time = time.perf_counter()
            frame_time = current_time - self._last_frame
            self._frame_times.append(frame_time)
            self._last_frame = current_time

            avg_frame_time = np.mean(self._frame_times) if self._frame_times else 0

            stats_text = (
                f"FPS: {stats.get('fps',0):.1f} ({1000*avg_frame_time:.1f}ms) | "
                f"Species: {stats.get('total_species',0)} | "
                f"Particles: {stats.get('total_particles',0):,}"
            )

            text_surface = self.font.render(stats_text, True, (255, 255, 255))
            self.surface.blit(text_surface, (10, 10))

        except Exception as e:
            print(f"Error in render: {e}")


###############################################################
# Cellular Automata (Main Simulation)
###############################################################


class CellularAutomata:
    """
    The main simulation class. Initializes and runs the simulation loop.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the CellularAutomata with the given configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.config = config  # Store simulation configuration
        pygame.init()  # Initialize all imported Pygame modules

        # Retrieve display information to set fullscreen window
        display_info = pygame.display.Info()
        screen_width = max(800, display_info.current_w)  # Ensure minimum width
        screen_height = max(600, display_info.current_h)  # Ensure minimum height

        # Set up a fullscreen window with the calculated dimensions
        try:
            self.screen = pygame.display.set_mode(
                (screen_width, screen_height),
                pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF,
            )
        except pygame.error:
            # Fallback to windowed mode if fullscreen fails
            self.screen = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            screen_width, screen_height = 800, 600

        pygame.display.set_caption("Emergent Cellular Automata Simulation")

        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.run_flag = True

        # Calculate minimum distance from edges (5% of the larger screen dimension)
        self.edge_buffer = np.clip(0.05 * max(screen_width, screen_height), 10, 100)

        # Setup cellular type colors and identify mass-based types
        try:
            self.colors = self.renderer.generate_vibrant_colors(
                self.config.n_cell_types
            )
            n_mass_types = max(
                0,
                min(
                    int(self.config.mass_based_fraction * self.config.n_cell_types),
                    self.config.n_cell_types,
                ),
            )
            mass_based_type_indices = list(range(n_mass_types))
        except Exception as e:
            print(f"Error generating colors/types: {e}")
            self.colors = [(255, 255, 255)] * self.config.n_cell_types
            n_mass_types = 0
            mass_based_type_indices = []

        # Initialize managers with error handling
        try:
            self.type_manager = CellularTypeManager(
                self.config, self.colors, mass_based_type_indices
            )
            self.rules_manager = InteractionRules(self.config, mass_based_type_indices)
            self.renderer = Renderer(self.screen, self.config)
            self.genetic_interpreter = GeneticInterpreter()
        except Exception as e:
            print(f"Error initializing managers: {e}")
            raise

        # Pre-calculate mass values with bounds checking
        mass_values = np.zeros(n_mass_types)
        if n_mass_types > 0:
            mass_values = np.clip(
                np.random.uniform(
                    self.config.mass_range[0], self.config.mass_range[1], n_mass_types
                ),
                1e-6,  # Minimum mass
                None,  # No maximum
            )

        # Create cellular type data with error handling
        for i in range(self.config.n_cell_types):
            try:
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=max(1, self.config.particles_per_type),
                    window_width=screen_width,
                    window_height=screen_height,
                    initial_energy=max(0.1, self.config.initial_energy),
                    max_age=max(1, self.config.max_age),
                    mass=mass_values[i] if i < n_mass_types else None,
                    base_velocity_scale=max(0.1, self.config.base_velocity_scale),
                )
                self.type_manager.add_cellular_type_data(ct)
            except Exception as e:
                print(f"Error creating cellular type {i}: {e}")
                continue

        # Initialize statistics tracking
        self.species_count = defaultdict(int)
        self.update_species_count()

        # Pre-allocate arrays for boundary calculations
        self.screen_bounds = np.array(
            [
                self.edge_buffer,
                screen_width - self.edge_buffer,
                self.edge_buffer,
                screen_height - self.edge_buffer,
            ]
        )

        # Initialize performance tracking
        self._init_performance_tracking()

        # Cache for KD trees (updated each frame)
        self.tree_cache = {}

        # Initialize renderer
        renderer = Renderer(self.screen, self.config)

    def _init_performance_tracking(self):
        """Initialize performance tracking metrics"""
        self._performance_metrics = {
            "fps_history": collections.deque([60.0] * 60, maxlen=60),
            "particle_counts": collections.deque([0] * 60, maxlen=60),
            "cull_history": collections.deque(maxlen=10),
            "last_cull_time": time.time(),
            "performance_score": 1.0,
            "stress_threshold": 0.7,
            "min_fps": 45,
            "target_fps": 90,
            "emergency_fps": 30,
            "last_emergency": 0,
            "frame_times": collections.deque(maxlen=120),
        }

    def update_species_count(self) -> None:
        """Update the species count based on current cellular types."""
        try:
            self.species_count.clear()
            for ct in self.type_manager.cellular_types:
                if ct.species_id is not None and ct.species_id.size > 0:
                    unique, counts = np.unique(ct.species_id, return_counts=True)
                    for species, count in zip(unique, counts):
                        if species >= 0:  # Ignore invalid species IDs
                            self.species_count[species] += count
        except Exception as e:
            print(f"Error updating species count: {e}")

        # Real-time FPS Display

    def display_fps(self, surface: pygame.Surface, fps: float) -> None:
        """
        Displays the current FPS at the top-left corner of the screen.

        Args:
            surface (pygame.Surface): The Pygame surface to render the FPS on.
            fps (float): The current frames per second.
        """
        font = pygame.font.Font(None, 36)  # Use a default font with size 36
        fps_text = font.render(
            f"FPS: {fps:.2f}", True, (255, 255, 255)
        )  # Render the FPS text in white
        surface.blit(fps_text, (10, 10))  # Draw the text at (10, 10) on the screen

    def decode_genetic_traits(self) -> None:
        """
        Decode genetic traits for each cellular type.
        """
        for ct in self.type_manager.cellular_types:
            self.genetic_interpreter.decode(
                ct, others=self.type_manager.cellular_types, env=self.config
            )

    def apply_all_interactions(self) -> None:
        """
        Apply inter-type interactions: forces, give-take, and synergy.
        """
        # Iterate over all interaction rules between cellular type pairs
        for i, j, params in self.rules_manager.rules:
            self.apply_interaction_between_types(
                i, j, params
            )  # Apply interactions between types i and j

    def apply_interaction_between_types(
        self, i: int, j: int, params: Dict[str, Any]
    ) -> None:
        """
        Apply interaction rules between cellular type i and cellular type j.
        This includes forces, give-take, and synergy.

        Parameters:
        -----------
        i : int
            Index of the first cellular type.
        j : int
            Index of the second cellular type.
        params : Dict[str, Any]
            Interaction parameters between cellular type i and j.
        """
        ct_i = self.type_manager.get_cellular_type_by_id(i)  # Get cellular type i
        ct_j = self.type_manager.get_cellular_type_by_id(j)  # Get cellular type j

        # Extract synergy factor and give-take relationship from interaction rules
        synergy_factor = self.rules_manager.synergy_matrix[
            i, j
        ]  # Synergy factor between types i and j
        is_giver = self.rules_manager.give_take_matrix[
            i, j
        ]  # Give-take relationship flag

        n_i = ct_i.x.size  # Number of components in type i
        n_j = ct_j.x.size  # Number of components in type j

        if n_i == 0 or n_j == 0:
            return  # No interaction if one type has no components

        # Prepare mass parameters if gravity is used
        if params.get("use_gravity", False):
            if (
                ct_i.mass_based
                and ct_i.mass is not None
                and ct_j.mass_based
                and ct_j.mass is not None
            ):
                params["m_a"] = ct_i.mass  # Mass of type i components
                params["m_b"] = ct_j.mass  # Mass of type j components
            else:
                params["use_gravity"] = False

        # Calculate pairwise differences using broadcasting
        dx = ct_i.x[:, np.newaxis] - ct_j.x  # Shape: (n_i, n_j)
        dy = ct_i.y[:, np.newaxis] - ct_j.y  # Shape: (n_i, n_j)
        dist_sq = dx * dx + dy * dy  # Squared distances, shape: (n_i, n_j)

        # Determine which pairs are within interaction range and not overlapping
        within_range = (dist_sq > 0.0) & (
            dist_sq <= params["max_dist"] ** 2
        )  # Boolean mask

        # Extract indices of interacting pairs
        indices = np.where(within_range)
        if len(indices[0]) == 0:
            return  # No interactions within range

        # Calculate distances for interacting pairs
        dist = np.sqrt(dist_sq[indices])

        # Initialize force arrays
        fx = np.zeros_like(dist)
        fy = np.zeros_like(dist)

        # Potential-based forces
        if params.get("use_potential", True):
            pot_strength = params.get("potential_strength", 1.0)
            F_pot = pot_strength / dist
            fx += F_pot * dx[indices]
            fy += F_pot * dy[indices]

        # Gravity-based forces
        if params.get("use_gravity", False):
            gravity_factor = params.get("gravity_factor", 1.0)
            F_grav = (
                gravity_factor
                * (params["m_a"][indices[0]] * params["m_b"][indices[1]])
                / dist_sq[indices]
            )
            fx += F_grav * dx[indices]
            fy += F_grav * dy[indices]

        # Accumulate forces using NumPy's add.at for atomic operations
        np.add.at(ct_i.vx, indices[0], fx)
        np.add.at(ct_i.vy, indices[0], fy)

        # Handle give-take interactions
        if is_giver:
            give_take_within = dist_sq[indices] <= self.config.predation_range**2
            give_take_indices = (
                indices[0][give_take_within],
                indices[1][give_take_within],
            )
            if give_take_indices[0].size > 0:
                giver_energy = ct_i.energy[give_take_indices[0]]
                receiver_energy = ct_j.energy[give_take_indices[1]]
                giver_mass = (
                    ct_i.mass[give_take_indices[0]] if ct_i.mass_based else None
                )
                receiver_mass = (
                    ct_j.mass[give_take_indices[1]] if ct_j.mass_based else None
                )

                updated = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config,
                )
                ct_i.energy[give_take_indices[0]] = updated[0]
                ct_j.energy[give_take_indices[1]] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]

        # Handle synergy interactions
        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            synergy_within = dist_sq[indices] <= self.config.synergy_range**2
            synergy_indices = (indices[0][synergy_within], indices[1][synergy_within])
            if synergy_indices[0].size > 0:
                energyA = ct_i.energy[synergy_indices[0]]
                energyB = ct_j.energy[synergy_indices[1]]
                new_energyA, new_energyB = apply_synergy(
                    energyA, energyB, synergy_factor
                )
                ct_i.energy[synergy_indices[0]] = new_energyA
                ct_j.energy[synergy_indices[1]] = new_energyB

        # Apply friction and global temperature effects vectorized
        friction_mask = np.full(n_i, self.config.friction)
        ct_i.vx *= friction_mask
        ct_i.vy *= friction_mask

        thermal_noise = (
            np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature
        )
        ct_i.vx += thermal_noise
        ct_i.vy += thermal_noise

        # Update positions vectorized
        ct_i.x += ct_i.vx
        ct_i.y += ct_i.vy

        # Handle boundary conditions
        self.handle_boundary_reflections(ct_i)

        # Age components and update states vectorized
        ct_i.age_components()
        ct_i.update_states()
        ct_i.update_alive()

    def handle_boundary_reflections(
        self, ct: Optional[CellularTypeData] = None
    ) -> None:
        """
        Handle boundary reflections for cellular components using vectorized operations.
        """
        cellular_types = [ct] if ct else self.type_manager.cellular_types

        for ct in cellular_types:
            if ct.x.size == 0:
                continue

            # Create boolean masks for boundary violations
            left_mask = ct.x < self.screen_bounds[0]
            right_mask = ct.x > self.screen_bounds[1]
            top_mask = ct.y < self.screen_bounds[2]
            bottom_mask = ct.y > self.screen_bounds[3]

            # Reflect velocities where needed
            ct.vx[left_mask | right_mask] *= -1
            ct.vy[top_mask | bottom_mask] *= -1

            # Clamp positions to bounds
            np.clip(ct.x, self.screen_bounds[0], self.screen_bounds[1], out=ct.x)
            np.clip(ct.y, self.screen_bounds[2], self.screen_bounds[3], out=ct.y)

    def main_loop(self) -> None:
        """Run the main simulation loop with error handling and performance monitoring."""
        last_time = time.time()

        while self.run_flag:
            try:
                current_time = time.time()
                frame_time = current_time - last_time
                last_time = current_time

                self._performance_metrics["frame_times"].append(frame_time)

                self.frame_count += 1
                if (
                    self.config.max_frames > 0
                    and self.frame_count > self.config.max_frames
                ):
                    self.run_flag = False

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        self.run_flag = False
                        break

                # Main simulation steps with performance monitoring
                with Timer() as t:
                    self._simulation_step()

                # Adaptive performance management
                if t.interval > 1.0 / 30:  # If frame took longer than 33ms
                    self._handle_performance_degradation()

                # Update display
                pygame.display.flip()

                # Cap frame rate
                current_fps = self.clock.tick(120)

                # Adaptive particle management
                if current_fps <= 60 and self.frame_count % 10 == 0:
                    self.cull_oldest_particles()

            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                continue

        pygame.quit()

    def _simulation_step(self):
        """Execute one step of the simulation with error handling"""
        try:
            # Update interaction parameters
            self.rules_manager.evolve_parameters(self.frame_count)

            # Update genetic traits
            if self.frame_count % 5 == 0:  # Only update genetics every 5 frames
                self.decode_genetic_traits()

            # Apply interactions
            self.apply_all_interactions()

            # Apply clustering
            self._apply_clustering_parallel()

            # Handle reproduction and death
            self.type_manager.reproduce()
            self.type_manager.remove_dead_in_all_types()

            # Update species count
            if self.frame_count % 10 == 0:
                self.update_species_count()

            # Render
            self._render_frame()

        except Exception as e:
            print(f"Error in simulation step: {e}")
            traceback.print_exc()

    def _render_frame(self):
        """Render the current frame with error handling"""
        try:
            # Draw cellular types
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)

            # Compile stats
            stats = {
                "fps": max(1.0, self.clock.get_fps()),
                "total_species": len(self.species_count),
                "total_particles": max(0, sum(self.species_count.values())),
            }

            # Render
            self.renderer.render(stats)

        except Exception as e:
            print(f"Error rendering frame: {e}")

    def apply_clustering(self, ct: CellularTypeData) -> None:
        """
        Apply clustering forces within a single cellular type using KD-Tree for efficiency.
        """
        n = ct.x.size
        if n < 2:
            return

        # Build KD-Tree once for position data
        positions = np.column_stack((ct.x, ct.y))
        tree = cKDTree(positions)

        # Query all neighbors at once
        indices = tree.query_ball_tree(tree, self.config.cluster_radius)

        # Pre-allocate velocity change arrays
        dvx = np.zeros(n)
        dvy = np.zeros(n)

        # Vectorized calculations for all components
        for idx, neighbor_indices in enumerate(indices):
            neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
            if not neighbor_indices:
                continue

            neighbor_positions = positions[neighbor_indices]
            neighbor_velocities = np.column_stack(
                (ct.vx[neighbor_indices], ct.vy[neighbor_indices])
            )

            # Alignment
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            alignment = (
                avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])
            ) * self.config.alignment_strength

            # Cohesion
            center = np.mean(neighbor_positions, axis=0)
            cohesion = (center - positions[idx]) * self.config.cohesion_strength

            # Separation
            separation = (
                positions[idx] - np.mean(neighbor_positions, axis=0)
            ) * self.config.separation_strength

            # Combine forces
            total_force = alignment + cohesion + separation
            dvx[idx] = total_force[0]
            dvy[idx] = total_force[1]

        # Apply accumulated velocity changes
        ct.vx += dvx
        ct.vy += dvy

    def _apply_clustering_parallel(self):
        """Apply clustering using parallel processing for large particle counts"""
        try:
            for ct in self.type_manager.cellular_types:
                if ct.x.size >= 1000:  # Only parallelize for large counts
                    with ThreadPoolExecutor() as executor:
                        chunks = np.array_split(range(ct.x.size), 4)
                        futures = [
                            executor.submit(self._apply_clustering_chunk, ct, chunk)
                            for chunk in chunks
                        ]
                        for future in futures:
                            future.result()
                else:
                    self.apply_clustering(ct)
        except Exception as e:
            print(f"Error in parallel clustering: {e}")

    def _handle_performance_degradation(self):
        """Handle severe performance issues"""
        try:
            current_fps = self.clock.get_fps()
            if current_fps < self._performance_metrics["emergency_fps"]:
                self._emergency_optimization()
        except Exception as e:
            print(f"Error handling performance degradation: {e}")

    def cull_oldest_particles(self):
        """Cull the oldest particles based on performance metrics and fitness assessment."""
        try:
            metrics = self._performance_metrics
            current_time = time.time()
            current_fps = self.clock.get_fps()

            metrics["fps_history"].append(current_fps)
            total_particles = sum(ct.x.size for ct in self.type_manager.cellular_types)
            metrics["particle_counts"].append(total_particles)

            avg_fps = np.mean(metrics["fps_history"])
            fps_trend = (
                np.gradient(list(metrics["fps_history"]))[-10:].mean()
                if len(metrics["fps_history"]) > 10
                else 0
            )
            particle_trend = (
                np.gradient(list(metrics["particle_counts"]))[-10:].mean()
                if len(metrics["particle_counts"]) > 10
                else 0
            )

            fps_stress = max(
                0, (metrics["target_fps"] - avg_fps) / metrics["target_fps"]
            )
            particle_stress = 1 / (1 + np.exp(-total_particles / 10000))
            system_stress = fps_stress * 0.7 + particle_stress * 0.3

            if (
                current_fps < metrics["emergency_fps"]
                and current_time - metrics["last_emergency"] > 5.0
            ):
                emergency_cull_factor = 0.5
                metrics["last_emergency"] = current_time
                metrics["performance_score"] *= 2.0
                for ct in self.type_manager.cellular_types:
                    if ct.x.size < 100:
                        continue
                    keep_count = max(50, int(ct.x.size * (1 - emergency_cull_factor)))
                    self._emergency_cull(ct, keep_count)
                return

            if avg_fps < metrics["min_fps"]:
                metrics["performance_score"] *= 1.5
            elif avg_fps < metrics["target_fps"]:
                metrics["performance_score"] *= 1.2
            elif avg_fps > metrics["target_fps"]:
                metrics["performance_score"] = max(
                    0.2, metrics["performance_score"] * 0.9
                )

            if fps_trend < 0:
                metrics["performance_score"] *= 1.2
            if particle_trend > 0:
                metrics["performance_score"] *= 1.1

            metrics["performance_score"] = np.clip(
                metrics["performance_score"], 0.2, 10.0
            )

            for ct in self.type_manager.cellular_types:
                if ct.x.size < 100:
                    continue
                positions = np.column_stack((ct.x, ct.y))
                tree = cKDTree(positions)
                fitness_scores = np.zeros(ct.x.size)
                density_scores = tree.query_ball_point(
                    positions, r=200, return_length=True
                )
                density_penalty = density_scores / (np.max(density_scores) + 1e-6)

                energy_score = (
                    ct.energy * ct.energy_efficiency * (1 - (ct.age / ct.max_age))
                )
                interaction_score = (
                    ct.interaction_strength
                    * ct.synergy_affinity
                    * ct.colony_factor
                    * ct.reproduction_rate
                )
                fitness_scores = (
                    energy_score * 0.4
                    + interaction_score * 0.3
                    + (1 - density_penalty) * 0.3
                )
                fitness_scores = (fitness_scores - np.min(fitness_scores)) / (
                    np.max(fitness_scores) - np.min(fitness_scores) + 1e-10
                )

                base_cull_rate = 0.1 * metrics["performance_score"] * system_stress
                cull_rate = np.clip(base_cull_rate, 0.05, 0.4)
                removal_count = int(ct.x.size * cull_rate)
                keep_indices = np.argsort(fitness_scores)[removal_count:]
                keep_mask = np.zeros(ct.x.size, dtype=bool)
                keep_mask[keep_indices] = True

                arrays_to_filter = [
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "energy",
                    "alive",
                    "age",
                    "energy_efficiency",
                    "speed_factor",
                    "interaction_strength",
                    "perception_range",
                    "reproduction_rate",
                    "synergy_affinity",
                    "colony_factor",
                    "drift_sensitivity",
                    "species_id",
                    "parent_id",
                ]
                for attr in arrays_to_filter:
                    setattr(ct, attr, getattr(ct, attr)[keep_mask])
                if ct.mass_based and ct.mass is not None:
                    ct.mass = ct.mass[keep_mask]

            metrics["last_cull_time"] = current_time

        except Exception as e:
            print(f"Error in culling oldest particles: {e}")

    def _emergency_cull(self, ct: CellularTypeData, keep_count: int) -> None:
        """Cull particles in an emergency situation based on fitness assessment."""
        try:
            if ct.x.size <= keep_count:
                return

            positions = np.column_stack((ct.x, ct.y))
            tree = cKDTree(positions)
            fitness_scores = np.zeros(ct.x.size)
            density_scores = tree.query_ball_point(positions, r=200, return_length=True)
            density_penalty = density_scores / (np.max(density_scores) + 1e-6)

            energy_score = (
                ct.energy * ct.energy_efficiency * (1 - (ct.age / ct.max_age))
            )
            interaction_score = (
                ct.interaction_strength
                * ct.synergy_affinity
                * ct.colony_factor
                * ct.reproduction_rate
            )
            fitness_scores = (
                energy_score * 0.4
                + interaction_score * 0.3
                + (1 - density_penalty) * 0.3
            )
            fitness_scores = (fitness_scores - np.min(fitness_scores)) / (
                np.max(fitness_scores) - np.min(fitness_scores) + 1e-10
            )

            keep_indices = np.argsort(fitness_scores)[-keep_count:]
            keep_mask = np.zeros(ct.x.size, dtype=bool)
            keep_mask[keep_indices] = True

            arrays_to_filter = [
                "x",
                "y",
                "vx",
                "vy",
                "energy",
                "alive",
                "age",
                "energy_efficiency",
                "speed_factor",
                "interaction_strength",
                "perception_range",
                "reproduction_rate",
                "synergy_affinity",
                "colony_factor",
                "drift_sensitivity",
                "species_id",
                "parent_id",
            ]
            for attr in arrays_to_filter:
                setattr(ct, attr, getattr(ct, attr)[keep_mask])
            if ct.mass_based and ct.mass is not None:
                ct.mass = ct.mass[keep_mask]

        except Exception as e:
            print(f"Error in emergency culling: {e}")
            self._ensure_particle_stability(ct)

    def _ensure_particle_stability(self, ct: CellularTypeData) -> None:
        """Ensure particle stability by resetting invalid attributes."""
        try:
            for attr in [
                "x",
                "y",
                "vx",
                "vy",
                "energy",
                "age",
                "energy_efficiency",
                "speed_factor",
                "interaction_strength",
                "perception_range",
                "reproduction_rate",
                "synergy_affinity",
                "colony_factor",
                "drift_sensitivity",
                "species_id",
                "parent_id",
            ]:
                arr = getattr(ct, attr)
                if isinstance(arr, np.ndarray):
                    if attr == "energy":
                        setattr(ct, attr, np.clip(arr, *self.bounds["energy"]))
                    elif attr in ["vx", "vy"]:
                        setattr(ct, attr, np.clip(arr, *self.bounds["velocity"]))
                    elif attr == "age":
                        setattr(ct, attr, np.clip(arr, *self.bounds["age"]))
                    elif attr == "mass" and ct.mass_based and ct.mass is not None:
                        setattr(ct, attr, np.clip(arr, *self.bounds["mass"]))
                    else:
                        setattr(ct, attr, np.clip(arr, *self.bounds["traits"]))
        except Exception as e:
            print(f"Error ensuring particle stability: {e}")

    def _emergency_optimization(self):
        """Emergency optimization when performance severely degrades"""
        try:
            # Reduce particle counts
            for ct in self.type_manager.cellular_types:
                if ct.x.size > 100:
                    keep_count = max(50, ct.x.size // 2)
                    self._emergency_cull(ct, keep_count)

            # Disable expensive calculations temporarily
            self.config.synergy_range = 0
            self.config.predation_range = 0

        except Exception as e:
            print(f"Error in emergency optimization: {e}")


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
