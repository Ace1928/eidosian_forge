#!/usr/bin/env python3
"""
Irrational Pi Visualizer – Single Cell with Adjustable Multi-Joint Arms
-----------------------------------------------------------------------
This simulation shows a single high-quality cell (centered on the screen)
with multiple arms. Each arm is composed of a configurable number of segments
that rotate relative to each other. A persistent neon trace is drawn from
the tip of each arm.

Use the UI controls on the right to adjust parameters in real time:
  • Number of arms (1–2400 when symmetry off; effectively doubled when symmetry ON)
  • Number of segments per arm (1–5000)
  • Arm length (pixels; 10–100000)
  • Base speed (0.001–10.0 radians/frame)
  • Speed multiplier (0.1–2000.0)
  • "Synchronized Speed" toggle
  • "Symmetry" toggle

Press ESC to exit.
"""

import sys
import math
import colorsys
import pygame
import pygame_gui
from typing import Tuple, List, Dict, Union, cast, TypedDict
from pygame_gui.elements import UIHorizontalSlider, UILabel, UIButton, UIPanel


# =============================================================================
# Configuration Constants
# =============================================================================
class SimulationParams(TypedDict):
    """Type definition for simulation parameters."""

    arms_count: int
    num_segments: int
    arm_length: float
    base_speed: float
    multiplier: float
    hue_cycle_period: int
    sync_speed: bool
    symmetry: bool
    trace_surface_scale: float
    frames_per_update: int


DEFAULT_PARAMS: SimulationParams = {
    "arms_count": 4,
    "num_segments": 2,
    "arm_length": 50.0,
    "base_speed": 0.05,
    "multiplier": math.pi,
    "hue_cycle_period": 360,
    "sync_speed": False,
    "symmetry": False,
    "trace_surface_scale": 0.8,
    "frames_per_update": 1,
}

UI_CONFIG = {
    "panel_width": 300,
    "slider_height": 25,
    "slider_spacing": 50,
    "slider_start_y": 20,
    "button_height": 35,
}


# =============================================================================
# Core Utility Functions
# =============================================================================
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Constrain value between minimum and maximum bounds."""
    return max(min_val, min(value, max_val))


def hsv_to_rgb(h: float = 0.0, s: float = 1.0, v: float = 1.0) -> Tuple[int, int, int]:
    """Convert HSV color (0.0–1.0) to RGB tuple (0–255) with validation."""
    h = clamp(h, 0.0, 1.0)
    s = clamp(s, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def compute_hue_value(
    frame_count: int, hue_cycle_period: int, arm_offset: int
) -> float:
    """Calculate normalized hue value with temporal and spatial offset."""
    validated_period = max(1, hue_cycle_period)
    return ((frame_count + arm_offset) % validated_period) / validated_period


def compute_color(
    frame_count: int = 0, hue_cycle_period: int = 360, arm_offset: int = 0
) -> Tuple[int, int, int]:
    """Generate cycling neon color with temporal and spatial offset."""
    hue = compute_hue_value(frame_count, hue_cycle_period, arm_offset)
    return hsv_to_rgb(hue, 1.0, 1.0)


# =============================================================================
# Simulation Core Components
# =============================================================================
class ArmState:
    """Container for arm segment state and dynamics."""

    def __init__(self, num_segments: int):
        self.angles: List[float] = [0.0] * num_segments
        self.speeds: List[float] = [0.0] * num_segments
        self.last_tip: Tuple[float, float] = (0.0, 0.0)


class SimulationState:
    """Data container for simulation runtime state."""

    def __init__(self):
        self.frame_count: int = 0
        self.trace_surface: pygame.Surface = None
        self.trace_center: Tuple[int, int] = (0, 0)
        self.arms: List[ArmState] = []


class SimulationParameters:
    """Validated simulation parameters with type safety."""

    def __init__(self, **kwargs):
        self.arms_count = max(1, kwargs.get("arms_count", DEFAULT_PARAMS["arms_count"]))
        self.num_segments = max(
            1, kwargs.get("num_segments", DEFAULT_PARAMS["num_segments"])
        )
        self.arm_length = max(
            10.0, kwargs.get("arm_length", DEFAULT_PARAMS["arm_length"])
        )
        self.base_speed = max(
            0.001, kwargs.get("base_speed", DEFAULT_PARAMS["base_speed"])
        )
        self.multiplier = max(
            0.1, kwargs.get("multiplier", DEFAULT_PARAMS["multiplier"])
        )
        self.hue_cycle_period = max(
            1, kwargs.get("hue_cycle_period", DEFAULT_PARAMS["hue_cycle_period"])
        )
        self.sync_speed = bool(kwargs.get("sync_speed", DEFAULT_PARAMS["sync_speed"]))
        self.symmetry = bool(kwargs.get("symmetry", DEFAULT_PARAMS["symmetry"]))
        self.trace_scale = clamp(
            kwargs.get("trace_surface_scale", DEFAULT_PARAMS["trace_surface_scale"]),
            0.1,
            1.0,
        )
        self.frames_per_update = max(
            1, kwargs.get("frames_per_update", DEFAULT_PARAMS["frames_per_update"])
        )


class SimulationEngine:
    """Core simulation logic and state management."""

    def __init__(self, screen_size: Tuple[int, int], params: SimulationParameters):
        self.screen_size = screen_size
        self.params = params
        self.state = SimulationState()
        self.initialize_simulation()
        self.update_counter: int = 0

    def initialize_simulation(self) -> None:
        """Set up initial simulation state."""
        self.state.frame_count = 0
        self.state.trace_surface = self.create_trace_surface()
        self.state.trace_center = (
            self.state.trace_surface.get_width() // 2,
            self.state.trace_surface.get_height() // 2,
        )
        self.generate_arms()
        self.update_counter = 0

    def create_trace_surface(self) -> pygame.Surface:
        """Create resizable trace surface with current settings."""
        size = int(min(self.screen_size) * self.params.trace_scale)
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))
        return surface

    def generate_arms(self) -> None:
        """Initialize arm geometry and dynamics."""
        self.state.arms.clear()
        effective_arms = (
            self.params.arms_count * 2
            if self.params.symmetry
            else self.params.arms_count
        )

        for arm_idx in range(effective_arms):
            arm = ArmState(self.params.num_segments)
            is_mirrored = self.params.symmetry and arm_idx >= self.params.arms_count
            base_idx = arm_idx - self.params.arms_count if is_mirrored else arm_idx

            arm.angles = self.calculate_initial_angles(base_idx, is_mirrored)
            arm.speeds = self.generate_speed_profile(base_idx)
            arm.last_tip = self.compute_arm_tip(arm.angles)

            self.state.arms.append(arm)

    def calculate_initial_angles(self, base_idx: int, mirrored: bool) -> List[float]:
        """Calculate initial arm angles with symmetry handling."""
        angle = (2 * math.pi * base_idx / self.params.arms_count) + (
            math.pi if mirrored else 0
        )
        return [angle] + [0.0] * (self.params.num_segments - 1)

    def generate_speed_profile(self, arm_idx: int) -> List[float]:
        """Generate speed profile for arm segments."""
        if self.params.sync_speed:
            return [self.params.base_speed] * self.params.num_segments

        offset = arm_idx * 0.005
        return [self.params.base_speed + offset] + [
            (self.params.base_speed * self.params.multiplier) + offset
        ] * (self.params.num_segments - 1)

    def compute_arm_tip(self, angles: List[float]) -> Tuple[float, float]:
        """Calculate arm tip position from segment angles."""
        cx, cy = map(float, self.state.trace_center)
        segment_len = self.params.arm_length / self.params.num_segments
        total_angle = 0.0

        for angle in angles:
            total_angle += angle
            cx += segment_len * math.cos(total_angle)
            cy += segment_len * math.sin(total_angle)

        return (cx, cy)

    def update_arm_kinematics(self) -> None:
        """Update arm segment angles and positions."""
        self.state.frame_count += 1

        for arm in self.state.arms:
            for seg_idx in range(self.params.num_segments):
                arm.angles[seg_idx] += arm.speeds[seg_idx]

    def update_trace_drawing(self) -> None:
        """Update trace paths on the drawing surface, only if update counter reaches threshold"""
        if self.update_counter >= self.params.frames_per_update:
            for arm_idx, arm in enumerate(self.state.arms):
                new_tip = self.compute_arm_tip(arm.angles)
                color = compute_color(
                    self.state.frame_count, self.params.hue_cycle_period, arm_idx * 20
                )
                pygame.draw.line(
                    self.state.trace_surface, color, arm.last_tip, new_tip, 2
                )
                arm.last_tip = new_tip
            self.update_counter = 0  # reset the counter
        else:
            self.update_counter += 1

    def handle_resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize events."""
        self.screen_size = new_size
        self.state.trace_surface = self.create_trace_surface()
        self.state.trace_center = (
            self.state.trace_surface.get_width() // 2,
            self.state.trace_surface.get_height() // 2,
        )
        self.reset_trace()

    def reset_trace(self) -> None:
        """Clear trace surface and reset arm positions."""
        self.state.trace_surface.fill((0, 0, 0, 0))
        for arm in self.state.arms:
            arm.last_tip = self.compute_arm_tip(arm.angles)


# =============================================================================
# UI Components
# =============================================================================
class UIFactory:
    """Factory class for creating UI elements."""

    @staticmethod
    def create_slider(
        manager: pygame_gui.UIManager,
        container: UIPanel,
        param: str,
        value_range: Tuple[float, float],
        format_str: str,
        position: Tuple[int, int],
        width: int,
        height: int,
        current_value: float,
    ) -> Tuple[UIHorizontalSlider, UILabel]:
        """Create a slider with associated label."""
        slider = UIHorizontalSlider(
            relative_rect=pygame.Rect(position[0], position[1], width, height),
            start_value=current_value,
            value_range=value_range,
            manager=manager,
            container=container,
            object_id=f"#{param}_slider",
        )
        label = UILabel(
            pygame.Rect(position[0], position[1] + height, width, 20),
            format_str.format(current_value),
            manager,
            container=container,
            object_id=f"#{param}_label",
        )
        return slider, label

    @staticmethod
    def create_toggle_button(
        manager: pygame_gui.UIManager,
        container: UIPanel,
        text: str,
        position: Tuple[int, int],
        size: Tuple[int, int],
        object_id: str,
    ) -> UIButton:
        """Create a toggle button with specified properties."""
        return UIButton(
            pygame.Rect(position[0], position[1], size[0], size[1]),
            text,
            manager,
            container=container,
            object_id=object_id,
        )


class UIManager:
    """Manage UI elements and state."""

    def __init__(self, screen_size: Tuple[int, int]):
        self.panel_width = UI_CONFIG["panel_width"]
        self.manager = self.initialize_ui_manager(screen_size)
        self.elements: Dict[
            str, Union[UIHorizontalSlider, UILabel, UIButton, UIPanel]
        ] = {}
        self.params = DEFAULT_PARAMS.copy()

        self.create_ui_layout(screen_size)

    def initialize_ui_manager(
        self, screen_size: Tuple[int, int]
    ) -> pygame_gui.UIManager:
        """Initialize UI manager with theme fallback."""
        try:
            return pygame_gui.UIManager(screen_size, "theme.json")
        except FileNotFoundError:
            return pygame_gui.UIManager(screen_size)

    def create_ui_layout(self, screen_size: Tuple[int, int]) -> None:
        """Create complete UI layout with all components."""
        self.create_main_panel(screen_size)
        self.create_control_elements()

    def create_main_panel(self, screen_size: Tuple[int, int]) -> None:
        """Create main UI panel container."""
        panel_rect = pygame.Rect(
            screen_size[0] - self.panel_width, 0, self.panel_width, screen_size[1]
        )
        self.elements["panel"] = UIPanel(
            panel_rect,
            manager=self.manager,
            object_id="#main_panel",
        )

    def create_control_elements(self) -> None:
        """Create all interactive control elements."""
        y_pos = UI_CONFIG["slider_start_y"]
        panel = cast(UIPanel, self.elements["panel"])

        # Create parameter sliders
        slider_width = self.panel_width - 20
        self.create_slider_group(
            panel=panel,
            y_start=y_pos,
            parameters=[
                ("arms_count", (1, 2400), "Arms: {}"),
                ("num_segments", (1, 5000), "Segments: {}"),
                ("arm_length", (10, 100000), "Length: {}"),
                ("base_speed", (0.001, 10.0), "Speed: {:.3f}"),
                ("multiplier", (0.1, 2000.0), "Multiplier: {:.3f}"),
            ],
        )

        # Create toggle buttons
        y_pos += len(UI_CONFIG) * (
            UI_CONFIG["slider_height"] + UI_CONFIG["slider_spacing"]
        )
        self.create_toggle_buttons(panel, y_pos)

    def create_slider_group(
        self,
        panel: UIPanel,
        y_start: int,
        parameters: List[Tuple[str, Tuple[float, float], str]],
    ) -> None:
        """Create a group of sliders for related parameters."""
        y_pos = y_start
        for param, value_range, format_str in parameters:
            self.elements[f"{param}_slider"], self.elements[f"{param}_label"] = (
                UIFactory.create_slider(
                    manager=self.manager,
                    container=panel,
                    param=param,
                    value_range=value_range,
                    format_str=format_str,
                    position=(10, y_pos),
                    width=self.panel_width - 20,
                    height=UI_CONFIG["slider_height"],
                    current_value=self.params[param],
                )
            )
            y_pos += UI_CONFIG["slider_height"] + UI_CONFIG["slider_spacing"]

    def create_toggle_buttons(self, panel: UIPanel, y_pos: int) -> None:
        """Create toggle buttons for boolean parameters."""
        btn_size = (self.panel_width - 20, UI_CONFIG["button_height"])

        self.elements["sync_btn"] = UIFactory.create_toggle_button(
            manager=self.manager,
            container=panel,
            text=f"Sync: {'ON' if self.params['sync_speed'] else 'OFF'}",
            position=(10, y_pos),
            size=btn_size,
            object_id="#sync_btn",
        )

        self.elements["symmetry_btn"] = UIFactory.create_toggle_button(
            manager=self.manager,
            container=panel,
            text=f"Symmetry: {'ON' if self.params['symmetry'] else 'OFF'}",
            position=(10, y_pos + UI_CONFIG["button_height"] + 10),
            size=btn_size,
            object_id="#symmetry_btn",
        )

    def get_current_parameters(self) -> SimulationParams:
        """Retrieve current parameters from UI elements."""
        return {
            "arms_count": int(
                cast(
                    UIHorizontalSlider, self.elements["arms_count_slider"]
                ).get_current_value()
            ),
            "num_segments": int(
                cast(
                    UIHorizontalSlider, self.elements["num_segments_slider"]
                ).get_current_value()
            ),
            "arm_length": float(
                cast(
                    UIHorizontalSlider, self.elements["arm_length_slider"]
                ).get_current_value()
            ),
            "base_speed": float(
                cast(
                    UIHorizontalSlider, self.elements["base_speed_slider"]
                ).get_current_value()
            ),
            "multiplier": float(
                cast(
                    UIHorizontalSlider, self.elements["multiplier_slider"]
                ).get_current_value()
            ),
            "hue_cycle_period": DEFAULT_PARAMS["hue_cycle_period"],
            "sync_speed": self.params["sync_speed"],
            "symmetry": self.params["symmetry"],
            "trace_surface_scale": DEFAULT_PARAMS["trace_surface_scale"],
            "frames_per_update": DEFAULT_PARAMS["frames_per_update"],
        }

    def handle_resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize events for UI elements."""
        self.manager.set_window_resolution(new_size)
        cast(UIPanel, self.elements["panel"]).kill()
        self.create_ui_layout(new_size)


# =============================================================================
# Main Application
# =============================================================================
class ApplicationState:
    """Container for application runtime state."""

    def __init__(self):
        self.running = True
        self.screen_size = (1280, 720)
        self.previous_params = DEFAULT_PARAMS.copy()


def initialize_systems() -> Tuple[pygame.Surface, SimulationEngine, UIManager]:
    """Initialize core application systems."""
    pygame.init()
    pygame.display.set_caption("Irrational Pi Visualizer")
    screen = pygame.display.set_mode(ApplicationState().screen_size, pygame.RESIZABLE)
    params = SimulationParameters(**DEFAULT_PARAMS)
    simulation = SimulationEngine(screen.get_size(), params)
    ui = UIManager(screen.get_size())
    return screen, simulation, ui


def handle_events(
    event: pygame.event.Event, ui: UIManager, simulation: SimulationEngine
) -> bool:
    """Process system and UI events."""
    if event.type == pygame.QUIT or (
        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
    ):
        return False

    if event.type == pygame.VIDEORESIZE:
        handle_resize(event, ui, simulation)

    if event.type == pygame_gui.UI_BUTTON_PRESSED:
        handle_button_press(event, ui)

    ui.manager.process_events(event)
    return True


def handle_resize(
    event: pygame.event.Event, ui: UIManager, simulation: SimulationEngine
) -> None:
    """Handle window resize event."""
    new_size = (event.w, event.h)
    pygame.display.set_mode(new_size, pygame.RESIZABLE)
    simulation.handle_resize(new_size)
    ui.handle_resize(new_size)


def handle_button_press(event: pygame.event.Event, ui: UIManager) -> None:
    """Handle UI button press events."""
    if event.ui_object_id == "#sync_btn":
        ui.params["sync_speed"] = not ui.params["sync_speed"]
        event.ui_element.set_text(f"Sync: {'ON' if ui.params['sync_speed'] else 'OFF'}")
    elif event.ui_object_id == "#symmetry_btn":
        ui.params["symmetry"] = not ui.params["symmetry"]
        event.ui_element.set_text(
            f"Symmetry: {'ON' if ui.params['symmetry'] else 'OFF'}"
        )


def update_simulation(
    simulation: SimulationEngine,
    ui: UIManager,
    current_params: SimulationParams,
    previous_params: SimulationParams,
) -> None:
    """Update simulation state if parameters changed."""
    if current_params != previous_params:
        simulation.params = SimulationParameters(**current_params)
        simulation.initialize_simulation()
        previous_params.clear()
        previous_params.update(current_params)

    simulation.update_arm_kinematics()
    simulation.update_trace_drawing()


def render_frame(
    screen: pygame.Surface, simulation: SimulationEngine, ui: UIManager
) -> None:
    """Render complete application frame."""
    screen.fill((30, 30, 30))

    # Draw simulation elements
    screen_center = (simulation.screen_size[0] // 2, simulation.screen_size[1] // 2)
    trace_pos = (
        screen_center[0] - simulation.state.trace_center[0],
        screen_center[1] - simulation.state.trace_center[1],
    )
    screen.blit(simulation.state.trace_surface, trace_pos)
    pygame.draw.circle(screen, (255, 0, 0), screen_center, 5)

    # Draw UI elements
    ui.manager.draw_ui(screen)
    pygame.display.update()


def main() -> None:
    """Main application entry point."""
    screen, simulation, ui = initialize_systems()
    clock = pygame.time.Clock()
    state = ApplicationState()

    while state.running:
        time_delta = clock.tick(60) / 1000.0

        # Event processing
        for event in pygame.event.get():
            state.running = handle_events(event, ui, simulation)

        # System updates
        ui.manager.update(time_delta)
        current_params = ui.get_current_parameters()
        update_simulation(simulation, ui, current_params, state.previous_params)

        # Frame rendering
        render_frame(screen, simulation, ui)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
