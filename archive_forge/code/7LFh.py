from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
import random


class Game(ShowBase):
    def __init__(self):
        super().__init__()

        # Initialize the window, camera, and other settings
        self.set_background_color(0.1, 0.1, 0.1, 1)
        self.cam.set_pos(0, -30, 10)
        self.cam.look_at(0, 0, 0)

        # Initialize lighting within the game environment
        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color(Vec4(0.2, 0.2, 0.2, 1))
        ambient_light_node = self.render.attach_new_node(ambient_light)
        self.render.set_light(ambient_light_node)

        point_light = PointLight("point_light")
        point_light.set_color(Vec4(0.9, 0.9, 0.9, 1))
        point_light_node = self.render.attach_new_node(point_light)
        point_light_node.set_pos(10, -20, 20)
        self.render.set_light(point_light_node)

        # Configure the Bullet physics world
        self.world = BulletWorld()
        self.world.set_gravity(Vec3(0, 0, -9.81))

        # Configure collision handling mechanisms
        self.traverser = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()

        # Construct the ground element
        self.create_ground()

        # Construct the player sphere
        self.create_player()

        # Register the player's collision node with the collision traverser
        self.traverser.add_collider(self.player_np, self.pusher)
        self.pusher.add_in_pattern("%fn-into-%in")
        self.pusher.add_out_pattern("%fn-out-%in")

        # Construct obstacles within the game
        self.create_obstacles()

        # Schedule regular updates
        self.task_mgr.add(self.update, "update")

    def create_ground(self):
        shape = BulletBoxShape(Vec3(10, 10, 1))
        body = BulletRigidBodyNode("Ground")
        body.add_shape(shape)
        np = self.render.attach_new_node(body)
        np.set_pos(0, 0, -2)
        self.world.attach_rigid_body(body)

    def create_player(self):
        shape = BulletSphereShape(1)
        body = BulletRigidBodyNode("Player")
        body.set_mass(1.0)
        body.add_shape(shape)
        self.player_np = self.render.attach_new_node(body)
        self.player_np.set_pos(0, 0, 2)
        self.world.attach_rigid_body(body)

        # Define the collision sphere for the player
        coll_node = CollisionNode("player")
        coll_node.add_solid(CollisionSphere(0, 0, 0, 1))
        coll_np = self.player_np.attach_new_node(coll_node)
        self.traverser.add_collider(coll_np, self.pusher)

    def create_obstacles(self):
        for _ in range(10):
            x, y, z = random.uniform(-8, 8), random.uniform(-8, 8), 0
            shape = BulletBoxShape(Vec3(1, 1, 1))
            body = BulletRigidBodyNode("Box")
            body.add_shape(shape)
            np = self.render.attach_new_node(body)
            np.set_pos(x, y, z)
            self.world.attach_rigid_body(body)

    def update(self, task):
        dt = globalClock.get_dt()
        self.world.do_physics(dt)
        return task.cont


game = Game()
game.run()
