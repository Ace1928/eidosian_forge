#!/usr/bin/env python3
"""
A refined, production-style demonstration of a procedurally generated multi-segmented,
multi-limbed creature learning to move in 2D under gravity with PyTorch and Pygame.

Differences / Improvements from previous versions:
    1) Multi-segment limbs (configurable number of segments per limb).
    2) Simple collision handling between segments (bounding-circle overlap resolution).
    3) Each joint has two muscle activations (push/pull) and one "max firing rate" scalar,
       so the net torque = (push - pull) * max_force. This prevents wild spasms and
       simulates push/pull muscle groups with a learnable relaxation limit.

Features Retained:
    - On-policy Policy Gradient with short rollouts and discounted returns.
    - Numeric robustness (clamping infinities/NaNs), gradient clipping, error handling.
    - Self-contained single-file code that can be run directly.
"""
import logging
import math
import random
import signal
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# -----------------------------------------------------------------------------
# LOGGING CONFIG
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS
# -----------------------------------------------------------------------------

# Screen / Visualization
SCREEN_WIDTH: int = 800
SCREEN_HEIGHT: int = 600
GROUND_LEVEL: int = 300  # For drawing the ground line at this y-pixel

# Physics
GRAVITY: float = 9.8  # Acceleration due to gravity (m/s^2)
TIME_STEP: float = 0.05  # Seconds per physics step

# Creature structure
NUM_SEGMENTS: int = 3  # Body segments in a chain
NUM_LIMBS: int = 4  # Number of limbs to attach to random body segments
NUM_LIMB_SEGMENTS: int = 2  # Segments per limb chain
SEGMENT_LENGTH: float = 40.0
SEGMENT_WIDTH: float = 10.0
LIMB_LENGTH: float = 30.0
LIMB_WIDTH: float = 8.0
JOINT_LIMIT: float = math.radians(45)  # Each joint rotates +/- 45° from rest

# RL / training
HIDDEN_SIZE: int = 128
LEARNING_RATE: float = 1e-3
GAMMA: float = 0.99
ROLLOUT_LENGTH: int = 20  # Steps per mini-trajectory
ROLLIN_STEPS_PER_SECOND: int = 30  # Pygame "FPS"

# Reward
FORWARD_REWARD_SCALE: float = 1.0
ALIVE_BONUS: float = 0.01

# Pygame
pygame.init()
FONT: pygame.font.Font = pygame.font.SysFont("Arial", 16)

# Numeric safety
np.seterr(divide="ignore", invalid="ignore")


# -----------------------------------------------------------------------------
# POLICY NETWORK
# -----------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    A feed-forward policy network for continuous action spaces.

    Each joint has 3 action outputs: push, pull, max_force.
    The final net torque = (push - pull) * max_force.

    The network outputs (mean, log_std) for each action dimension,
    and samples from a Normal distribution.

    Args:
        state_size: Number of state dimensions.
        action_size: Number of action dimensions.
        hidden_size: Size of hidden layers.

    Attributes:
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
        mean_head: Output layer for action means.
        log_std: Learnable log standard deviation.
    """

    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 128
    ) -> None:
        """Initialize the policy network with appropriate layers and weights."""
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))  # learnable log std

        # Layer initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Tensor of state values [batch_size, state_size].

        Returns:
            mean: Action mean values.
            log_std: Log standard deviations.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the Normal distribution defined by mean and std.

        Args:
            state: Tensor of state values [batch_size, state_size].

        Returns:
            action: Sampled actions [batch_size, action_dim].
            log_prob: Log probability of sampled actions [batch_size].
            entropy: Entropy of the distribution [batch_size].
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        action = dist.rsample()  # Reparameterized sampling for backprop
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy


# -----------------------------------------------------------------------------
# CREATURE & JOINT CLASSES
# -----------------------------------------------------------------------------
class CreatureSegment:
    """
    Represents a segment (or limb piece) of the creature.

    Each segment has position, velocity, angle, angular velocity,
    and a bounding circle for collision detection.

    Args:
        length: Length of the segment.
        width: Width of the segment.
        init_pos: Initial (x, y) position.
        init_angle: Initial angle in radians.

    Attributes:
        length: Length of the segment.
        width: Width of the segment.
        radius: Bounding circle radius for collision detection.
        x, y: Position coordinates.
        vx, vy: Velocity components.
        angle: Current angle in radians.
        angular_velocity: Current angular velocity.
        on_ground: Whether the segment is in contact with the ground.
    """

    def __init__(
        self,
        length: float,
        width: float,
        init_pos: Tuple[float, float],
        init_angle: float = 0.0,
    ) -> None:
        """Initialize a creature segment with physical properties."""
        self.length: float = length
        self.width: float = width

        # For collision checks, we use a bounding circle with radius ~ half diagonal
        # but to keep it simpler, let's just use half the length as radius
        # (or a max of length/2, width/2).
        self.radius: float = max(self.length / 2.0, self.width / 2.0)

        # 2D physics state
        self.x: float = init_pos[0]
        self.y: float = init_pos[1]
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.angle: float = init_angle
        self.angular_velocity: float = 0.0

        self.on_ground: bool = False

    def apply_physics(self, gravity: float, dt: float) -> None:
        """
        Apply physics simulation to the segment.

        Integrates velocity with gravity, handles ground collisions and friction.

        Args:
            gravity: Gravity acceleration constant.
            dt: Time step for integration.
        """
        self.vy += gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.angular_velocity * dt

        # Ground collision (y=0 is ground)
        if self.y > 0.0:
            self.on_ground = False
        else:
            self.y = 0.0
            self.vy *= -0.5  # Some bounce/damping
            if abs(self.vy) < 1.0:
                self.vy = 0.0
            self.on_ground = True

        # Ground friction
        if self.on_ground:
            self.vx *= 0.95  # Simple friction

    def render(
        self,
        screen: pygame.Surface,
        center_x: float,
        center_y: float,
        scale: float = 1.0,
    ) -> None:
        """
        Draw the segment as a rotated rectangle on the screen.

        Args:
            screen: Pygame surface to draw on.
            center_x: X-coordinate of the center point reference.
            center_y: Y-coordinate of the center point reference.
            scale: Scaling factor for rendering.
        """
        rect_length = int(self.length * scale)
        rect_width = int(self.width * scale)

        screen_x = center_x + int(self.x * scale)
        screen_y = center_y - int(self.y * scale)

        surface = pygame.Surface((rect_length, rect_width), pygame.SRCALPHA)
        surface.fill((100, 100, 200, 180))  # color + alpha
        rotated_surface = pygame.transform.rotate(surface, -math.degrees(self.angle))

        offset_x = rotated_surface.get_width() // 2
        offset_y = rotated_surface.get_height() // 2
        screen.blit(rotated_surface, (screen_x - offset_x, screen_y - offset_y))


class Joint:
    """
    Connects two CreatureSegments with constraints and muscle actuation.

    The child segment is constrained to remain within angle_limit of
    the rest_angle relative to the parent's angle.

    Args:
        parent: Parent segment this joint is attached to.
        child: Child segment controlled by this joint.
        rest_angle: Neutral/resting angle for this joint.
        angle_limit: Maximum angular deviation allowed from rest_angle.

    Attributes:
        parent: Parent segment this joint connects to.
        child: Child segment controlled by this joint.
        rest_angle: Neutral angle that defines the joint's rest position.
        angle_limit: Maximum allowed angular deviation from rest position.
        current_angle: Current relative angle between segments.
    """

    def __init__(
        self,
        parent: CreatureSegment,
        child: CreatureSegment,
        rest_angle: float,
        angle_limit: float = JOINT_LIMIT,
    ) -> None:
        """Initialize a joint between two segments with physical constraints."""
        self.parent: CreatureSegment = parent
        self.child: CreatureSegment = child
        self.rest_angle: float = rest_angle
        self.angle_limit: float = angle_limit
        self.current_angle: float = 0.0

    def apply_action(
        self, push: float, pull: float, max_force: float, dt: float
    ) -> None:
        """
        Apply muscle forces to the joint based on action values.

        Push and pull are antagonistic forces, with max_force limiting their effect.

        Args:
            push: Push force (must be >= 0).
            pull: Pull force (must be >= 0).
            max_force: Maximum force multiplier (must be >= 0).
            dt: Time step for integration.
        """
        net_torque = (push - pull) * max_force
        self.child.angular_velocity += net_torque * dt

        desired_abs_angle = self.parent.angle + self.rest_angle
        angle_diff = self.child.angle - desired_abs_angle

        # Clamp angle difference within +/- angle_limit
        if angle_diff < -self.angle_limit:
            self.child.angle = desired_abs_angle - self.angle_limit
            self.child.angular_velocity = 0.0
        elif angle_diff > self.angle_limit:
            self.child.angle = desired_abs_angle + self.angle_limit
            self.child.angular_velocity = 0.0

        self.current_angle = self.child.angle - self.parent.angle

    def get_relative_angle(self) -> float:
        """
        Calculate the relative angle between child and parent.

        Returns:
            The difference between the current joint angle and the rest angle.
        """
        return (self.child.angle - self.parent.angle) - self.rest_angle


class Creature:
    """
    A multi-segment creature with body and limbs connected by joints.

    The creature consists of a main body chain and multiple limbs,
    each potentially having multiple segments. Joints connect these
    segments and collision resolution is handled for all segments.

    Args:
        num_segments: Number of body segments in the main chain.
        num_limbs: Number of limbs to attach to the body.
        seed: Random seed for reproducible creature generation.
        num_limb_segments: Number of segments per limb.

    Attributes:
        segments: List of all segments (body + limbs).
        joints: List of joints connecting the segments.
        body_segments: List of segments in the main body chain.
        limb_segments: List of segments in the limbs.
    """

    def __init__(
        self,
        num_segments: int,
        num_limbs: int,
        seed: Optional[int] = None,
        num_limb_segments: int = 2,
    ) -> None:
        """Initialize a creature with specified morphology."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.segments: List[CreatureSegment] = []
        self.joints: List[Joint] = []

        # 1) Create main body chain
        for i in range(num_segments):
            seg = CreatureSegment(
                length=SEGMENT_LENGTH,
                width=SEGMENT_WIDTH,
                init_pos=(i * SEGMENT_LENGTH, 0.0),
                init_angle=0.0,
            )
            self.segments.append(seg)

            # Connect each new segment to the previous via a joint
            if i > 0:
                j = Joint(
                    parent=self.segments[i - 1],
                    child=seg,
                    rest_angle=0.0,  # Horizontal chain by default
                )
                self.joints.append(j)

        self.body_segments: List[CreatureSegment] = self.segments[:num_segments]

        # 2) Create limbs, each with num_limb_segments in a chain
        for _ in range(num_limbs):
            # attach the first limb segment to a random body segment
            parent_index = random.randint(0, num_segments - 1)
            parent_seg = self.segments[parent_index]

            limb_angle = random.uniform(-math.pi, math.pi)

            # Build a small chain of 'num_limb_segments'
            prev_seg = None
            for s_idx in range(num_limb_segments):
                # Create limb segment
                limb_pos = (
                    parent_seg.x if s_idx == 0 else prev_seg.x,
                    parent_seg.y if s_idx == 0 else prev_seg.y,
                )
                seg = CreatureSegment(
                    length=LIMB_LENGTH,
                    width=LIMB_WIDTH,
                    init_pos=limb_pos,
                    init_angle=parent_seg.angle + limb_angle,
                )
                self.segments.append(seg)

                # Create joint between parent/prev_seg and this segment
                if s_idx == 0:
                    j = Joint(parent=parent_seg, child=seg, rest_angle=limb_angle)
                else:
                    j = Joint(parent=prev_seg, child=seg, rest_angle=0.0)
                self.joints.append(j)
                prev_seg = seg

        # List for all segments (body + limbs)
        self.limb_segments: List[CreatureSegment] = self.segments[
            num_segments:
        ]  # everything after body

    def step(self, action: np.ndarray, dt: float) -> None:
        """
        Advance the creature's physics simulation based on muscle actions.

        Each action triplet (push, pull, max_force) controls a joint,
        with net torque = (push - pull) * max_force.

        Args:
            action: Array of action values [3 * #joints].
            dt: Time step for physics integration.
        """
        # 1) Clean up any NaNs / Inf
        with np.errstate(invalid="ignore"):
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) For each joint, interpret (push, pull, max_force)
        num_joints = len(self.joints)
        if len(action) != 3 * num_joints:
            logger.warning("Action dimension mismatch! Expected 3 * #joints.")
            return  # skip

        idx = 0
        for j in range(num_joints):
            push_raw = action[idx]
            pull_raw = action[idx + 1]
            maxf_raw = action[idx + 2]
            idx += 3

            # push, pull, and max_force must be >= 0 for muscle model
            # We'll apply a softplus or ReLU clamp here:
            push = max(0.0, push_raw)
            pull = max(0.0, pull_raw)
            maxf = max(0.0, maxf_raw)

            self.joints[j].apply_action(push, pull, maxf, dt)

        # 3) Apply base physics to each segment
        for seg in self.segments:
            seg.apply_physics(GRAVITY, dt)

        # 4) Collision resolution among all segments (naive pairwise bounding-circle push)
        self._resolve_collisions()

    def _resolve_collisions(self) -> None:
        """
        Resolve collisions between segments using bounding circle push-apart.

        If distance < sum_of_radii, segments are pushed apart equally.
        """
        n = len(self.segments)
        for i in range(n):
            for j in range(i + 1, n):
                segA = self.segments[i]
                segB = self.segments[j]

                dx = segB.x - segA.x
                dy = segB.y - segA.y
                dist_sq = dx * dx + dy * dy
                min_dist = segA.radius + segB.radius

                if dist_sq < (min_dist * min_dist):
                    dist = math.sqrt(dist_sq)
                    if dist < 0.01:  # avoid division by zero
                        continue

                    # Push segments apart along the vector between centers
                    overlap = min_dist - dist
                    push_x = dx / dist * overlap * 0.5
                    push_y = dy / dist * overlap * 0.5

                    # Apply equal and opposite pushes
                    segA.x -= push_x
                    segA.y -= push_y
                    segB.x += push_x
                    segB.y += push_y

    def get_position(self) -> float:
        """
        Get the creature's approximate forward position.

        Returns:
            Average x-coordinate of all body segments.
        """
        return float(np.mean([seg.x for seg in self.body_segments]))

    def get_velocity(self) -> float:
        """
        Get the creature's approximate forward velocity.

        Returns:
            Average x-velocity of all body segments.
        """
        return float(np.mean([seg.vx for seg in self.body_segments]))

    def get_state(self) -> np.ndarray:
        """
        Get the creature's current state vector.

        Returns:
            Array containing relative angles and angular velocities
            for each joint [2 * #joints].
        """
        angles = []
        ang_vels = []
        for joint in self.joints:
            angles.append(joint.get_relative_angle())
            ang_vels.append(joint.child.angular_velocity)
        return np.concatenate([angles, ang_vels], axis=0)

    def render(
        self,
        screen: pygame.Surface,
        center_x: float,
        center_y: float,
        scale: float = 1.0,
    ) -> None:
        """
        Render the creature on the Pygame screen.

        Args:
            screen: Pygame surface to draw on.
            center_x: X-coordinate of center reference point.
            center_y: Y-coordinate of center reference point.
            scale: Scaling factor for rendering.
        """
        for seg in self.segments:
            seg.render(screen, center_x, center_y, scale)


# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
class Environment:
    """
    2D plane environment with a multi-segment creature.

    The creature attempts to move forward, with the camera centered
    horizontally on the creature's position.

    Args:
        num_segments: Number of body segments for the creature.
        num_limbs: Number of limbs to attach to the body.
        seed: Random seed for reproducible environment.

    Attributes:
        num_segments: Number of body segments in creature.
        num_limbs: Number of limbs attached to creature.
        seed: Random seed for reproducibility.
        creature: The creature instance being simulated.
        last_x_position: Previous x-position for reward calculation.
        t: Current simulation time.
    """

    def __init__(
        self,
        num_segments: int = NUM_SEGMENTS,
        num_limbs: int = NUM_LIMBS,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the environment with specified creature parameters."""
        self.num_segments: int = num_segments
        self.num_limbs: int = num_limbs
        self.seed: Optional[int] = seed

        self.creature: Creature = Creature(0, 0)  # Placeholder initialization
        self.last_x_position: float = 0.0
        self.t: float = 0.0
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the environment by regenerating the creature.

        Returns:
            Initial state observation vector.
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Random seed for creature morphology
        creature_seed = random.randint(0, 999999)
        self.creature = Creature(
            num_segments=self.num_segments,
            num_limbs=self.num_limbs,
            seed=creature_seed,
            num_limb_segments=NUM_LIMB_SEGMENTS,
        )
        self.last_x_position = self.creature.get_position()
        self.t = 0.0
        return self.creature.get_state()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Step the simulation forward using the provided action.

        Args:
            action: Action vector [3 * #joints] for all joints.

        Returns:
            Tuple containing:
                - next_state: Next state observation.
                - reward: Reward for this step.
                - done: Whether the episode is finished.
                - info: Additional information dictionary.
        """
        self.creature.step(action, TIME_STEP)
        self.t += TIME_STEP

        # Reward based on forward progress + alive bonus
        x_position = self.creature.get_position()
        progress = x_position - self.last_x_position
        self.last_x_position = x_position

        reward = ALIVE_BONUS + FORWARD_REWARD_SCALE * progress

        # done if creature goes too far back or if time exceeded
        done = (x_position < -100.0) or (self.t > 60.0)

        next_state = self.creature.get_state()
        return next_state, reward, done, {}

    def render(self, screen: pygame.Surface) -> None:
        """
        Render the environment with the creature.

        Args:
            screen: Pygame surface to draw on.
        """
        screen.fill((255, 255, 255))

        creature_x = self.creature.get_position()
        center_x = SCREEN_WIDTH // 2
        center_y = GROUND_LEVEL
        scale = 1.0

        # Draw ground
        pygame.draw.line(screen, (0, 0, 0), (0, center_y), (SCREEN_WIDTH, center_y), 2)

        # Render creature offset so that x=0 is at center_x
        self.creature.render(
            screen, center_x - int(creature_x * scale), center_y, scale
        )

    def get_state_size(self) -> int:
        """
        Get the dimensionality of the state space.

        Returns:
            Number of dimensions in the state vector.
        """
        return len(self.creature.get_state())

    def get_action_size(self) -> int:
        """
        Get the dimensionality of the action space.

        Returns:
            Number of dimensions in the action vector (3 * #joints).
        """
        num_joints = len(self.creature.joints)
        return 3 * num_joints


# -----------------------------------------------------------------------------
# POLICY GRADIENT AGENT
# -----------------------------------------------------------------------------
class PolicyGradientAgent:
    """
    A policy gradient agent that learns using on-policy rollouts.

    The agent adapts to the 3-output-per-joint action structure:
    (push, pull, max_force) for each joint.

    Args:
        state_size: Dimensionality of the state space.
        action_size: Dimensionality of the action space.
        hidden_size: Size of hidden layers in the policy network.
        lr: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.

    Attributes:
        gamma: Discount factor for future rewards.
        policy_network: Neural network representing the policy.
        optimizer: Optimizer for policy network training.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = HIDDEN_SIZE,
        lr: float = LEARNING_RATE,
        gamma: float = GAMMA,
    ) -> None:
        """Initialize the agent with policy network and optimizer."""
        self.gamma: float = gamma
        self.policy_network: PolicyNetwork = PolicyNetwork(
            state_size, action_size, hidden_size
        )
        self.optimizer: optim.Adam = optim.Adam(self.policy_network.parameters(), lr=lr)

    def get_action(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.

        Args:
            state: Current state observation.

        Returns:
            Tuple containing:
                - action: Action vector to execute in environment.
                - log_prob: Log probability of the sampled action.
                - entropy: Entropy of the action distribution.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # [1, state_size]
        action_t, log_prob_t, entropy_t = self.policy_network.sample_action(state_t)

        # For stepping in the environment, we detach
        action_np = action_t[0].detach().cpu().numpy()

        # Squeeze out the batch dimension for log_prob, entropy
        log_prob = log_prob_t.squeeze(0)
        entropy = entropy_t.squeeze(0)
        return action_np, log_prob, entropy

    def update(self, trajectory: List[Dict[str, torch.Tensor]]) -> None:
        """
        Perform a policy gradient update using a collected trajectory.

        Args:
            trajectory: List of transition dictionaries, each containing
                       'log_prob', 'entropy', and 'reward'.
        """
        if len(trajectory) == 0:
            return

        # 1) Discounted returns
        returns = []
        r = 0.0
        for step in reversed(trajectory):
            r = step["reward"] + self.gamma * r
            returns.insert(0, r)
        returns_t = torch.FloatTensor(returns)

        # 2) Gather log_probs and entropies
        log_probs = torch.stack([t["log_prob"] for t in trajectory])
        entropies = torch.stack([t["entropy"] for t in trajectory])

        # 3) Clean up any NaNs
        if not torch.isfinite(returns_t).all():
            logger.warning("Encountered non-finite returns. Clamping.")
            returns_t = torch.nan_to_num(returns_t, nan=0.0, posinf=0.0, neginf=0.0)

        # 4) Policy gradient loss = mean of [ -log_prob * return - 0.001 * entropy ]
        loss = -log_probs * returns_t - 0.001 * entropies
        loss = loss.mean()

        # 5) Update
        self.optimizer.zero_grad()
        try:
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
            self.optimizer.step()
        except RuntimeError as e:
            logger.error(f"Backward pass failed: {e}")
            logger.info("Skipping this update step.")


# -----------------------------------------------------------------------------
# SIGNAL HANDLING & UTILITIES
# -----------------------------------------------------------------------------
def setup_graceful_exit(
    agent: PolicyGradientAgent, rollout_buffer: List[Dict[str, torch.Tensor]]
) -> None:
    """
    Configure signal handlers for graceful program termination.

    Ensures that neural network updates are performed with any
    remaining data before exiting.

    Args:
        agent: Policy gradient agent to update.
        rollout_buffer: Buffer of collected transitions.
    """

    def signal_handler(sig: int, frame: Any) -> None:
        """
        Handle termination signals by processing remaining data and exiting.

        Args:
            sig: Signal number.
            frame: Current stack frame.
        """
        logger.info(
            "🧠 Caught termination signal! Saving your precious learning data..."
        )
        # Process remaining transitions - waste not, want not
        if len(rollout_buffer) > 0:
            logger.info(
                f"Processing final {len(rollout_buffer)} transitions before exit..."
            )
            agent.update(rollout_buffer)

        logger.info("👋 Neural networks tucked in safely. Goodbye!")
        pygame.quit()
        sys.exit(0)

    # Register for both SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Run the main training loop with visualization.

    Creates environment, agent, and runs the simulation with
    learning updates from collected rollouts.
    """
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(
        "Procedural Multi-Segmented Creature (Push/Pull + Collisions)"
    )

    clock = pygame.time.Clock()

    # Create environment
    env = Environment(num_segments=NUM_SEGMENTS, num_limbs=NUM_LIMBS, seed=None)
    state_size = env.get_state_size()
    action_size = env.get_action_size()

    # Agent
    agent = PolicyGradientAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=HIDDEN_SIZE,
        lr=LEARNING_RATE,
        gamma=GAMMA,
    )

    num_episodes = 10_000_000
    episode = 0

    state = env.reset()
    total_reward = 0.0
    rollout_buffer: List[Dict[str, torch.Tensor]] = []

    # Set up signal handling for graceful exit
    setup_graceful_exit(agent, rollout_buffer)

    running = True
    while running:
        clock.tick(ROLLIN_STEPS_PER_SECOND)

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get action from agent
        action, log_prob, entropy = agent.get_action(state)
        # Step environment
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Buffer for short rollouts
        rollout_buffer.append(
            {"log_prob": log_prob, "entropy": entropy, "reward": reward}
        )

        # Render
        env.render(screen)

        # Info text
        info_text = (
            f"Episode: {episode} | "
            f"StepReward: {reward:.3f} | "
            f"EpReward: {total_reward:.3f}"
        )
        text_surf = FONT.render(info_text, True, (0, 0, 0))
        screen.blit(text_surf, (10, 10))
        pygame.display.flip()

        # Advance state
        state = next_state

        # If we hit the rollout length, update
        if len(rollout_buffer) >= ROLLOUT_LENGTH:
            agent.update(rollout_buffer)
            rollout_buffer = []

        if done:
            logger.info(
                f"Episode {episode} finished with total reward {total_reward:.3f}"
            )
            # Final update
            if len(rollout_buffer) > 0:
                agent.update(rollout_buffer)
                rollout_buffer = []
            episode += 1
            state = env.reset()
            total_reward = 0.0

    # Process remaining transitions when game loop ends
    # This ensures we don't waste learning data when terminating early
    if len(rollout_buffer) > 0:
        logger.info("Processing final transitions before exit...")
        agent.update(rollout_buffer)

    pygame.quit()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
